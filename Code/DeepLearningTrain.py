import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
from DatasetPreparation import DatasetPreparation
import copy
import torchmetrics

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
        image = Image.open(f"/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/Images/{self.image_names[index]}.png").convert('L')
        #print(np.shape(image))
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomAffine(degrees=10),
            # transforms.RandomPerspective(distortion_scale=0.1),
            # #transforms.ElasticTransform(alpha=50, sigma=5.0),
            # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.3)),
            transforms.Grayscale(num_output_channels=3),
            # transforms.ColorJitter(brightness=(0.8,1.2), contrast=0.2, saturation=(0.1,0.2), hue=(-0.1,0.1)),
            transforms.ToTensor(),
            # #transforms.Normalize(mean=mean, std=std) #does not work with PIL image, uzywam PIL na import, trzeba simpleITK
        ])
        image = self.transform(image)
        targets = self.labels[index]

        # to_pil = transforms.ToPILImage()
        # pil_image = to_pil(image)

        # # Display the image using plt.imshow()
        # plt.imshow(pil_image)
        # plt.axis('off')
        # plt.show()
        return {
            'image':image,
            'label': torch.tensor(targets, dtype=torch.float32)}


# # class ResNetGrayscale(nn.Module):
# #     def __init__(self, num_classes):
# #         super(ResNetGrayscale, self).__init__()
# #         self.resnet = resnet50(progress=True, pretrained=True)
# #         self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
# #         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

# def model(pretrained):
#     model = resnet18(progress=True, pretrained=pretrained)
#     model.fc = nn.Linear(model.fc.in_features, 10)
#     return model

# def forward(self, x):
#     return self.resnet(x)


# def train(model, dataloader, optimizer, criterion, device, class_weights):
#     print('Training')
#     model.train()
#     train_running_loss = 0.0
#     train_predictions = []
#     train_ground_truth = []
#     counter = 0
    
#     for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
#         counter += 1
#         images, labels = data['image'].to(device), data['label'].to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         #print(outputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         train_running_loss += loss.item()
#         predicted_labels = (torch.sigmoid(outputs*class_weights) > 0.5).float()
#         train_predictions.append(predicted_labels)
#         train_ground_truth.append(labels)
    
#     train_loss = train_running_loss / len(dataloader)
#     train_predictions = torch.cat(train_predictions, dim=0)  # Concatenate the list of tensors
#     train_ground_truth = torch.cat(train_ground_truth, dim=0)  # Concatenate the list of tensors
#     train_accuracy = torchmetrics.classification.MultilabelAccuracy(num_labels=10, average="macro").to(device)
#     train_accuracy=train_accuracy(train_predictions,train_ground_truth)
#     f1 = torchmetrics.classification.MultilabelF1Score(num_labels=10, average='macro').to(device)
#     f1 = f1(train_predictions,train_ground_truth)

#     return train_loss, train_accuracy, f1


# def validate(model, dataloader, criterion, device, class_weights):
#     print('Validating')
#     model.eval()
#     val_running_loss = 0.0
#     val_predictions = []
#     val_ground_truth = []
#     counter = 0
    
#     with torch.no_grad():
#         for i, data in tqdm(enumerate(dataloader), total=int(len(valid_data)/dataloader.batch_size)):
#             counter += 1
#             images, labels = data['image'].to(device), data['label'].to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
            
#             val_running_loss += loss.item()
#             predicted_labels = (torch.sigmoid(outputs*class_weights) > 0.5).float()
#             val_predictions.append(predicted_labels)
#             val_ground_truth.append(labels)
    
#     val_loss = val_running_loss / len(dataloader)
#     val_predictions = torch.cat(val_predictions, dim=0)  # Concatenate the list of tensors
#     val_ground_truth = torch.cat(val_ground_truth, dim=0)  # Concatenate the list of tensors
#     val_accuracy = torchmetrics.classification.MultilabelAccuracy(num_labels=10, average="macro").to(device)
#     val_accuracy=val_accuracy(val_predictions,val_ground_truth)
#     f1 = torchmetrics.classification.MultilabelF1Score(num_labels=10, average='macro').to(device)
#     f1 = f1(val_predictions,val_ground_truth)
    
#     return val_loss, val_accuracy, f1


# ## To be activated when class or specie changes
# # desired_class = 'LL'
# # desired_specie = 'cane'
# # DatasetPreparation(desired_class,desired_specie)


# # Initialize the computation device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Set random seed for reproducibility
# torch.manual_seed(42)

# # Initialize the model
# #model = ResNetGrayscale(num_classes=10).to(device)
# model = model(pretrained=True).to(device)

# # Freeze all layers except for fc and avgpool initially
# for name, param in model.named_parameters():
#     if 'fc' in name or 'avgpool' in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

# # Define the block_names in the desired order for gradual unfreezing
# block_names = ['layer4.1','layer4.0', 'layer3.1','layer3.0', 'layer2.1', 'layer2.0', 'layer1.1', 'layer1.0', '']
# model_list = []
# # Gradual unfreezing loop
# for block_name in block_names:
    
#     # Unfreeze the current block's convolutional layers and their batch normalization layers
#     for b in range(2,0,-1):
#         for param_name, param in model.named_parameters():
#             if block_name in param_name and (f'conv{b}' in param_name or f'bn{b}' in param_name or 'downsample' in param_name):
#                 param.requires_grad = True
#         model_copy = copy.deepcopy(model) 
#         model_list.append(model_copy)
# model_list=model_list[5:6]

# counter=0
# for model in model_list:
#     counter+=1
#     print('COUNTER:',counter)
    
#     # ct = 0
#     # for child in model.children():
#     #     ct += 1
#     #     if ct < a:
#     #         for param in child.parameters():
#     #                 param.requires_grad = False

#     # Learning parameters
#     lr = 0.0001
#     epochs = 120
#     batch_size = 64
#     optimizer = optim.AdamW(model.parameters(), lr=lr)
#     class_weights = torch.tensor([1, 2.6, 3.7, 3.7, 4.0, 5.2, 10.4, 15.3, 18.1, 18.5]).to(device)  # to be adjucted based on amount of classes, in the future with dataframe counts
#     criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
#     #criterion = nn.BCEWithLogitsLoss()

#     # Read the training csv file
#     train_csv = pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_train.csv')
#     train_csv = dropzerolabels(train_csv)

#     val_csv = pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_val.csv')
#     val_csv = dropzerolabels(val_csv)

#     # Train dataset
#     train_data = ImageDataset(train_csv, train=True, test=False)
    
#     # # Create a WeightedRandomSampler to perform oversampling during training
#     # train_sampler = WeightedRandomSampler(class_weights, len(train_data), replacement=True)


#     # Validation dataset
#     valid_data = ImageDataset(val_csv, train=False, test=False)

#     # Train data loader, train sampler w celu oversamplingu
#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False) #, sampler=train_sampler

#     # Validation data loader
#     valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

#     # Start the training and validation
#     train_loss = []
#     valid_loss = []
#     train_accuracy = []
#     valid_accuracy = []
#     valid_f1_temp=0

#     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     train_log_dir = f'/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Temporary/Logs for layers/train {str(counter)}'
#     val_log_dir = f'/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Temporary/Logs for layers/val {str(counter)}'
#     train_summary_writer = tb.SummaryWriter(log_dir=train_log_dir)
#     val_summary_writer = tb.SummaryWriter(log_dir=val_log_dir)


#     for epoch in range(epochs):
#         print(f"Epoch {epoch+1} of {epochs}")
#         train_epoch_loss, train_epoch_accuracy, train_f1 = train(model, train_loader, optimizer, criterion, device,class_weights)
#         train_summary_writer.add_scalar('loss', train_epoch_loss, global_step=epoch)
#         train_summary_writer.add_scalar('accuracy', train_epoch_accuracy, global_step=epoch)
#         train_summary_writer.add_scalar('F1', train_f1, global_step=epoch)
#         valid_epoch_loss, valid_epoch_accuracy, valid_f1 = validate(model, valid_loader, criterion, device,class_weights)
#         val_summary_writer.add_scalar('loss', valid_epoch_loss, global_step=epoch)
#         val_summary_writer.add_scalar('accuracy', valid_epoch_accuracy, global_step=epoch)
#         val_summary_writer.add_scalar('F1', valid_f1, global_step=epoch)
#         train_loss.append(train_epoch_loss)
#         valid_loss.append(valid_epoch_loss)
#         train_accuracy.append(train_epoch_accuracy)
#         valid_accuracy.append(valid_epoch_accuracy)
#         print(f"Train Loss: {train_epoch_loss:.4f}, \tTrain Accuracy: {train_epoch_accuracy:.4f}, \tTrain F1: {train_f1:.4f}")
#         print(f"Val Loss: {valid_epoch_loss:.4f}, \tVal Accuracy: {valid_epoch_accuracy:.4f}, \tVal F1: {valid_f1:.4f}")
#         if valid_f1>valid_f1_temp:
#             # Save the trained model to disk
#             torch.save(model.state_dict(), f'Models based on f1/model, good mesophagus highest f1 {valid_f1} 07-08.pth')
#             valid_f1_temp=valid_f1


#     # Save the trained model to disk
#     torch.save(
#         model.state_dict()
#     , 'model, good mesophagus 07-08.pth')
