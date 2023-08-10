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
from DropZeroValueLabels import dropzerolabels
import torch.utils.tensorboard as tb
import tensorflow as tf
import datetime
import io
import seaborn as sns
from DatasetPreparation import DatasetPreparation
import copy
import pydicom

class ImageDataset(Dataset):
    def __init__(self, csv, train, test):
        self.csv = csv
        self.train = train
        self.test = test
        
        if self.train == True:
            self.image_names = list(self.csv[:]['FileName'])
            self.labels = np.array(self.csv.drop(['FileName', 'TAG'], axis=1))
            print(f"Number of training images: {len(self.image_names)}")

        # set the validation data images and labels
        elif self.train == False and self.test == False:
            self.image_names = list(self.csv[:]['FileName'])
            self.labels = np.array(self.csv.drop(['FileName', 'TAG'], axis=1))
            print(f"Number of validation images: {len(self.image_names)}")

        # set the test data images and labels, only last 10 images
        # this, we will use in a separate inference script
        elif self.test == True and self.train == False:
            self.image_names = list(self.csv[:]['FileName'])
            self.labels = np.array(self.csv.drop(['FileName', 'TAG'], axis=1))
            print(f"Number of test images: {len(self.image_names)}")

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        dicom_file_path = f"/OLD-DATA-STOR/X-RAY-Project/db/completeDatabase/completeData/{self.image_names[index]}"
        dicom_data = pydicom.dcmread(dicom_file_path)
        pixel_array = dicom_data.pixel_array

        # Normalize pixel values to the range [0, 1]
        image = pixel_array / np.max(pixel_array)

        # Convert to PIL Image format
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=30),
            transforms.RandomPerspective(distortion_scale=0.1),
            #transforms.ElasticTransform(alpha=50, sigma=5.0),
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), nie dziala dobrze
            #transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.01, 0.3)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            #transforms.Normalize(mean=mean, std=std)
        ])
                # Apply transformations to the PIL Image
        image = self.transform(pil_image)

        # Targets (you can modify this according to your dataset's labels)
        targets = self.labels[index]

        # Display the DICOM image
        plt.imshow(pil_image, cmap='gray')
        plt.title('Original DICOM Image')
        plt.show()

        # Display the transformed image
        pil_transformed_image = transforms.ToPILImage()(image)
        plt.imshow(pil_transformed_image, cmap='gray')
        plt.title('Transformed Image')
        plt.show()

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

# Initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
torch.manual_seed(42)

# Initialize the model
#model = ResNetGrayscale(num_classes=10).to(device)
model = model(pretrained=True).to(device)

# Freeze all layers except for fc and avgpool initially
for name, param in model.named_parameters():
    if 'fc' in name or 'avgpool' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Define the block_names in the desired order for gradual unfreezing
block_names = ['layer4.1','layer4.0', 'layer3.1','layer3.0', 'layer2.1', 'layer2.0', 'layer1.1', 'layer1.0', '']
model_list = []
# Gradual unfreezing loop
for block_name in block_names:
    
    # Unfreeze the current block's convolutional layers and their batch normalization layers
    for b in range(2,0,-1):
        for param_name, param in model.named_parameters():
            if block_name in param_name and (f'conv{b}' in param_name or f'bn{b}' in param_name or 'downsample' in param_name):
                param.requires_grad = True
        model_copy = copy.deepcopy(model) 
        model_list.append(model_copy)

counter=0
for model in model_list:
    counter+=1
    print('COUNTER:',counter)
    
    # ct = 0
    # for child in model.children():
    #     ct += 1
    #     if ct < a:
    #         for param in child.parameters():
    #                 param.requires_grad = False

    # Learning parameters
    lr = 0.0001
    epochs = 20
    batch_size = 32
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    class_weights = torch.tensor([1, 2.6, 3.7, 3.7, 4.0, 5.2, 10.4, 15.3, 18.1, 18.5]).to(device)  # to be adjucted based on amount of classes, in the future with dataframe counts
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    ## To be activated when class or specie changes
    # desired_class = 'LL'
    # desired_specie = 'cane'
    # DatasetPreparation(desired_class,desired_specie)


    # Read the training csv file
    train_csv = pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_train.csv')
    train_csv = dropzerolabels(train_csv)

    val_csv = pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_val.csv')
    val_csv = dropzerolabels(val_csv)

    # Train dataset
    train_data = ImageDataset(train_csv, train=True, test=False)

    # Validation dataset
    valid_data = ImageDataset(val_csv, train=False, test=False)

    # Train data loader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    # Validation data loader
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    # Start the training and validation
    train_loss = []
    valid_loss = []
    train_accuracy = []
    valid_accuracy = []

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Temporary/Logs for layers/train {str(counter)}'
    val_log_dir = f'/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Temporary/Logs for layers/val {str(counter)}'
    matrix_log_dir = f'/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Temporary/Logs for layers/Matrix {str(counter)}'
    f1_log_dir = f'/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Temporary/Logs for layers/f1 {str(counter)}'
    ROC_log_dir = f'/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Temporary/Logs for layers/ROC {str(counter)}'
    train_summary_writer = tb.SummaryWriter(log_dir=train_log_dir)
    val_summary_writer = tb.SummaryWriter(log_dir=val_log_dir)
    matrix_summary_writer = tb.SummaryWriter(log_dir=matrix_log_dir)
    f1_summary_writer = tb.SummaryWriter(log_dir=f1_log_dir)
    ROC_summary_writer = tb.SummaryWriter(log_dir=ROC_log_dir)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_accuracy = train(model, train_loader, optimizer, criterion, device)
        train_summary_writer.add_scalar('loss', train_epoch_loss, global_step=epoch)
        train_summary_writer.add_scalar('accuracy', train_epoch_accuracy, global_step=epoch)
        valid_epoch_loss, valid_epoch_accuracy = validate(model, valid_loader, criterion, device)
        val_summary_writer.add_scalar('loss', valid_epoch_loss, global_step=epoch)
        val_summary_writer.add_scalar('accuracy', valid_epoch_accuracy, global_step=epoch)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        valid_accuracy.append(valid_epoch_accuracy)
        print(f"Train Loss: {train_epoch_loss:.4f}, \tTrain Accuracy: {train_epoch_accuracy:.4f}")
        print(f"Validation Loss: {valid_epoch_loss:.4f}, \tValidation Accuracy: {valid_epoch_accuracy:.4f}")

                # Evaluation
        model.eval()
        model.to(device)  # Move the model to the GPU

        test_csv = pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_test.csv')
        test_csv = dropzerolabels(test_csv)

        # Evaluation dataset with test=True
        eval_data = ImageDataset(test_csv, train=False, test=True)
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


        plt.figure(figsize=(12, 8))
        for i in range(num_classes):
            plt.subplot(2, 5, i + 1)
            sns.heatmap(confusion[i], annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Class {i + 1}')
            plt.xticks([0.5, 1.5], [0, 1])
            plt.yticks([0.5, 1.5], [0, 1])
            plt.tight_layout()

        # Convert the plot to an image and log it to TensorBoard
        confusion_image = io.BytesIO()
        plt.savefig(confusion_image, format='png')
        plt.close()
        confusion_image = Image.open(confusion_image)

        # Convert the image to a numpy array and convert the format to CHW
        confusion_image_np = np.array(confusion_image)
        confusion_image_np = np.transpose(confusion_image_np, (2, 0, 1))  # Convert to CHW format

        # Add the image to TensorBoard using tf.summary.image
        matrix_summary_writer.add_image(f'Multi-Label Confusion Matrix  {str(counter)}', confusion_image_np, global_step=epoch)

    
        # Create a figure and axes for the F1 score plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, num_classes + 1), f1_score)
        plt.xlabel('Class Number')
        plt.ylabel('F1 Score')
        plt.title('F1 Score for each Class')
        plt.xticks(range(1, num_classes + 1), range(1, num_classes + 1))  # Set class numbers as x-axis labels
        plt.ylim(0, 1)  # Set the y-axis range (F1 score is between 0 and 1)
        plt.grid(axis='y')
        plt.tight_layout()

        # Convert the plot to an image and log it to TensorBoard
        f1_score_image = io.BytesIO()
        plt.savefig(f1_score_image, format='png')
        plt.close()
        f1_score_image = Image.open(f1_score_image)

        f1_score_image_np = np.array(f1_score_image)
        f1_score_image_np = np.transpose(f1_score_image_np, (2,0,1))
        f1_summary_writer.add_image(f'F1 Score {str(counter)}', f1_score_image_np, global_step=epoch)
    
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

        # Convert the plot to an image and log it to TensorBoard
        ROC_image = io.BytesIO()
        plt.savefig(ROC_image, format='png')
        plt.close()
        ROC_image = Image.open(ROC_image)

        # Convert the image to a numpy array and log it to TensorBoard using tf.summary.image
        ROC_image_np = np.array(ROC_image)
        ROC_image_np = np.transpose(ROC_image_np, (2,0,1))
        ROC_summary_writer.add_image(f'Multi-Label ROC  {str(counter)}', ROC_image_np, global_step=epoch)

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
