
# import torch
# import pandas as pd
# from torch import nn
# from torch.utils.data import DataLoader
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torchvision.models import resnet18, resnext50_32x4d
# from torchvision.models import resnet50, resnext101_32x8d
# from torchvision.models import efficientnet_b0, efficientnet_b4
# import pytorch_lightning as pl
# from torchmetrics import Accuracy
# from efficientnet_pytorch import EfficientNet
# from torchvision.datasets import ImageFolder
# from sklearn.model_selection import train_test_split


# dataset = ImageFolder('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Results/test dp', transform =  transforms.Compose([transforms.Grayscale()])) #https://discuss.pytorch.org/t/how-to-use-the-imagefolder-function-to-load-single-channel-image/3925/2

# print(dataset, '\n')


# train_dataset, test_dataset = train_test_split(
#     dataset, test_size=0.2, shuffle=True, random_state=42
# )

# test_dataset, val_dataset = train_test_split(
#     test_dataset, test_size=0.2, shuffle=True, random_state=42
# )

# print("Train dataset number of files: ", len(train_dataset))
# print("Validation dataset number of files: ", len(val_dataset))
# print("Test dataset number of files: ", len(test_dataset))


# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)



# class ClassificationModel(pl.LightningModule):
#     def __init__(self, model_name, num_classes):
#         super(ClassificationModel, self).__init__()
#         if model_name == 'resnet18':
#             self.model = resnet18(pretrained=True)
#             num_features = self.model.fc.in_features
#         elif model_name == 'resnext50':
#             self.model = resnext50_32x4d(pretrained=True)
#             num_features = self.model.fc.in_features
#         elif model_name == 'efficientnet':
#             self.model = EfficientNet.from_pretrained('efficientnet-b0')
#             num_features = self.model._fc.in_features

#         self.model.fc = nn.Linear(num_features, num_classes)
        
#         #self.accuracy = Accuracy()

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         images, labels = batch
#         outputs = self(images)
#         loss = nn.CrossEntropyLoss()(outputs, labels)
#         self.log('train_loss', loss)
#         self.log('train_acc', self.accuracy(outputs, labels))
#         return loss

#     def validation_step(self, batch, batch_idx):
#         images, labels = batch
#         outputs = self(images)
#         loss = nn.CrossEntropyLoss()(outputs, labels)
#         self.log('val_loss', loss)
#         self.log('val_acc', self.accuracy(outputs, labels))

#     def test_step(self, batch, batch_idx):
#         images, labels = batch
#         outputs = self(images)
#         loss = nn.CrossEntropyLoss()(outputs, labels)
#         self.log('test_loss', loss)
#         self.log('test_acc', self.accuracy(outputs, labels))

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

# # Initialize the model
# model = ClassificationModel(model_name='resnet18', num_classes=len(dataset.classes))

# # Initialize the PyTorch Lightning Trainer
# trainer = pl.Trainer(gpus=1, max_epochs=10)

# # Train the model
# trainer.fit(model, train_loader, val_loader)

# # Test the model
# trainer.test(model, test_loader)


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, resnet18
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import multilabel_confusion_matrix, classification_report, precision_recall_fscore_support, roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from DropZeroValueLabels import dropzerolabels
import torch.utils.tensorboard
import tensorflow as tf
import datetime
from DatasetPreparation import DatasetPreparation


class ImageDataset(Dataset):
    def __init__(self, csv, train, test):
        self.csv = csv
        self.train = train
        self.test = test
        self.all_image_names = csv[:]['FileName']
        self.all_labels = np.array(csv.drop(['FileName', 'TAG'], axis=1))
        
        self.train_indices, self.remaining_indices = train_test_split(
            np.arange(len(self.csv)), train_size=0.70, shuffle=True, random_state=42
        )
        self.val_indices, self.test_indices = train_test_split(
            self.remaining_indices, train_size=0.3, shuffle=True, random_state=42
        )
        
        #self.train_ratio = int(0.85 * len(self.csv))
        #self.valid_ratio = len(self.csv) - self.train_ratio
        
        if self.train == True:
            print(f"Number of training images: {len(self.train_indices)}")
            self.image_names = list(self.all_image_names.iloc[self.train_indices])
            self.labels = list(self.all_labels[self.train_indices])

        # set the validation data images and labels
        elif self.train == False and self.test == False:
            print(f"Number of validation images: {len(self.val_indices)}")
            self.image_names = list(self.all_image_names.iloc[self.val_indices])
            self.labels = list(self.all_labels[self.val_indices])

        # set the test data images and labels, only last 10 images
        # this, we will use in a separate inference script
        elif self.test == True and self.train == False:
            print(f"Number of test images: {len(self.test_indices)}")
            self.image_names = list(self.all_image_names.iloc[self.test_indices])
            self.labels = list(self.all_labels[self.test_indices])

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image = Image.open(f"/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/Images/{self.image_names[index]}.png").convert("L")
        #print(np.shape(image))
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=30),
            transforms.RandomPerspective(distortion_scale=0.1),
            #transforms.ElasticTransform(alpha=50, sigma=5.0),
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), nie dziala dobrze
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.01, 0.3)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        image = self.transform(image)
        targets = self.labels[index]

        return {
            'image':image,
            'label': torch.tensor(targets, dtype=torch.float32)}


# class ResNetGrayscale(nn.Module):
#     def __init__(self, num_classes):
#         super(ResNetGrayscale, self).__init__()
#         self.resnet = resnet50(progress=True, pretrained=True)
#         self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
def model(pretrained):
    model = resnet18(progress=True, pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def forward(self, x):
    return self.resnet(x)

# Set up TensorBoard writer
writer = torch.utils.tensorboard.SummaryWriter('logs')

def train(model, dataloader, optimizer, criterion, device):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_predictions = []
    train_ground_truth = []
    counter = 0
    
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        counter += 1
        images, labels = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        #print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_running_loss += loss.item()
        predicted_labels = (torch.sigmoid(outputs) > 0.7).float()
        train_predictions.append(predicted_labels.cpu().numpy())
        train_ground_truth.append(labels.cpu().numpy())
    
    train_loss = train_running_loss / len(dataloader)
    train_predictions = np.concatenate(train_predictions)
    train_ground_truth = np.concatenate(train_ground_truth)
    train_accuracy = accuracy_score(train_ground_truth, train_predictions)

    return train_loss, train_accuracy


def validate(model, dataloader, criterion, device):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_predictions = []
    val_ground_truth = []
    counter = 0
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(valid_data)/dataloader.batch_size)):
            counter += 1
            images, labels = data['image'].to(device), data['label'].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item()
            predicted_labels = (torch.sigmoid(outputs) > 0.7).float()
            val_predictions.append(predicted_labels.cpu().numpy())
            val_ground_truth.append(labels.cpu().numpy())
    
    val_loss = val_running_loss / len(dataloader)
    val_predictions = np.concatenate(val_predictions)
    val_ground_truth = np.concatenate(val_ground_truth)
    val_accuracy = accuracy_score(val_ground_truth, val_predictions)
    
    return val_loss, val_accuracy

# Close TensorBoard writer
writer.close()

# Initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
torch.manual_seed(42)

# Initialize the model
#model = ResNetGrayscale(num_classes=10).to(device)
model = model(pretrained=True).to(device)
ct = 0
for child in model.children():
    ct += 1
    if ct < 7:
        for param in child.parameters():
                param.requires_grad = False

# Learning parameters
lr = 0.002
epochs = 10
batch_size = 32
optimizer = optim.AdamW(model.parameters(), lr=lr)
class_weights = torch.tensor([1, 2.6, 3.7, 3.7, 4.0, 5.2, 10.4, 15.3, 18.1, 18.5]).to(device)  # to be adjucted based on amount of classes, in the future with dataframe counts
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

## To be activated when class or specie changes
# desired_class = 'LL'
# desired_specie = 'cane'
# DatasetPreparation(desired_class,desired_specie)


# Read the training csv file
train_csv = pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP.csv')
train_csv = dropzerolabels(train_csv)

# Train dataset
train_data = ImageDataset(train_csv, train=True, test=False)

# Validation dataset
valid_data = ImageDataset(train_csv, train=False, test=False)

# Train data loader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Validation data loader
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

# Start the training and validation
train_loss = []
valid_loss = []
train_accuracy = []
valid_accuracy = []

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = '/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Temporary/Logs tensorboard' + current_time + '/train'
test_log_dir = '/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Temporary/Logs tensorboard' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_accuracy = train(model, train_loader, optimizer, criterion, device)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_epoch_loss, step=epoch)
        tf.summary.scalar('accuracy', train_epoch_accuracy, step=epoch)
    valid_epoch_loss, valid_epoch_accuracy = validate(model, valid_loader, criterion, device)
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', valid_epoch_loss, step=epoch)
        tf.summary.scalar('accuracy', valid_epoch_accuracy, step=epoch)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    valid_accuracy.append(valid_epoch_accuracy)
    print(f"Train Loss: {train_epoch_loss:.4f}, \tTrain Accuracy: {train_epoch_accuracy:.4f}")
    print(f"Validation Loss: {valid_epoch_loss:.4f}, \tValidation Accuracy: {valid_epoch_accuracy:.4f}")

# Save the trained model to disk
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': criterion,
}, 'model.pth')

# Plot and save the train and validation line graphs
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()

# with torch.utils.tensorboard.SummaryWriter('logs') as writer:
#     model = resnet18(pretrained=True)
#     inputs = torch.randn(1, 3, 224, 224)
#     writer.add_graph(model, inputs)

# Evaluation
model.eval()
model.to(device)  # Move the model to the GPU

# Evaluation dataset with test=True
eval_data = ImageDataset(train_csv, train=False, test=True)
eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)

predictions = []
labels = []
outputssigmoid = []

with torch.no_grad():
    for data in tqdm(eval_loader):
        images, true_labels = data['image'].to(device), data['label'].to(device)
        outputs = model(images)
        outputs=torch.sigmoid(outputs)
        predicted_labels = (outputs > 0.7).float()

        outputssigmoid.append(outputs.cpu().numpy())
        predictions.append(predicted_labels.cpu().numpy())
        labels.append(true_labels.cpu().numpy())

# Concatenate predictions and labels after the loop to obtain the final arrays
outputssigmoid = np.concatenate(outputssigmoid, axis=0)
predictions = np.concatenate(predictions, axis=0)
labels = np.concatenate(labels, axis=0)

# Confusion Matrix
confusion = multilabel_confusion_matrix(labels, predictions)
print("shape",confusion.shape)  
# Calculate Precision, Recall, and F1 Score for each class individually
precision, recall, f1_score, _ = precision_recall_fscore_support(labels, predictions, average=None)

# Print the results for each class
num_classes = len(precision)
for i in range(num_classes):
    print(f"Class {i + 1} - Precision: {precision[i]}, Recall: {recall[i]}, F1 Score: {f1_score[i]}")

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(labels.ravel(), outputssigmoid.ravel())
auc = roc_auc_score(labels.ravel(), outputssigmoid.ravel())

# Print the evaluation results
print("Confusion Matrix:")
print(confusion)
print("\nClassification Report:")
#print(report)
print("\nF1 Score:", f1_score)
print("Precision:", precision)
print("Recall:", recall)
print("AUC:", auc)

# Plot ROC Curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()