import torch.utils.tensorboard as tb
import torch
import torchvision.models as models
import torch.nn as nn
from DropZeroValueLabels import dropzerolabels
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from DeepLearningTrain import ImageDataset
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import io
from PIL import Image
import seaborn as sns
from torchvision.utils import make_grid


figs_weighted=[]
figs_oversampled=[]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Code/Testing/end model, oversampled good mesophagus 08-08.pth'))
batch_size=64

model.to(device)  # Move the model to the GPU

val_csv = pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_test.csv')
val_csv = dropzerolabels(val_csv)

# Evaluation dataset with test=True
eval_data = ImageDataset(val_csv, train=False, test=False)
eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)

predictions = []
labels = []
outputssigmoid = []
number_of_classes=10

STEP=0.01 #how often the threshold is evaluated per class

#czyli dla klasy fora, szukac threshold, albo wektor albo 10 petli for
with torch.no_grad():
    for data in tqdm(eval_loader):
        images, true_labels = data['image'].to(device), data['label'].to(device)
        outputs = model(images)
        outputs=torch.sigmoid(outputs)

        outputssigmoid.append(outputs)
        labels.append(true_labels)

    outputssigmoid = torch.cat(outputssigmoid, dim=0)
    labels = torch.cat(labels, dim=0)
        
    for i in range(number_of_classes):
        print("Class:",i)
        matrix_log_dir = f'/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Temporary/Logs Threshold eval/Matrix Class{str(i)}'
        f1_log_dir = f'/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Temporary/Logs Threshold eval/f1 Class{str(i)}'
        ROC_log_dir = f'/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Temporary/Logs Threshold eval/ROC Class{str(i)}'
            
        matrix_summary_writer = tb.SummaryWriter(log_dir=matrix_log_dir)
        f1_summary_writer = tb.SummaryWriter(log_dir=f1_log_dir)
        ROC_summary_writer = tb.SummaryWriter(log_dir=ROC_log_dir)
        
        current_outputs = outputssigmoid[:, i]
        current_labels = labels[:, i]

        thresholds=np.arange(0,1,STEP)
        f1_scores_graph=[]

        for j in thresholds:
            #print("Threshold:",j)
            outputs_thresholded = current_outputs > j
            # Calculate Precision, Recall, and F1 Score for each class individually
            #precision, recall, f1_score, _ = precision_recall_fscore_support(current_labels, current_outputs, average=None)
            f1 = torchmetrics.classification.BinaryF1Score(multidim_average='global').to(device)
            f1 = f1(outputs_thresholded,current_labels)
            f1_summary_writer.add_scalar('F1', f1, global_step=j)
            #print(f1.cpu().numpy())
            f1_scores_graph.append(f1.cpu().numpy())
            Correct_label_count = sum(current_labels.cpu().numpy())

            # Confusion Matrix
            confusion = confusion_matrix(outputs_thresholded.cpu().numpy(), current_labels.cpu().numpy())
            confusion_image = torch.tensor(confusion).float().unsqueeze(0)  # Convert to a 1x2x2 tensor
            confusion_image = make_grid(confusion_image, nrow=1, normalize=True)  # Create a grid of 1x2 images
            # Log confusion matrix image to TensorBoard
            print(confusion)
            matrix_summary_writer.add_image(f'Multi-Label Confusion Matrix Class {str(i)}', confusion_image, global_step=j)
        
        plt.plot(thresholds,f1_scores_graph)
        plt.title(f"Positive cases:{Correct_label_count}")
        plt.show()


    # for w zakresie klas "klasa"
    #     for threshlolkd zmiana c0 0.1 lub 0.01 jako x
    #         predicted_labels = (torch.sigmoid(outputs[klasa]) > x).float()
    #         predictions.append(predicted_labels.cpu().numpy())

        # Concatenate predictions and labels after the loop to obtain the final arrays
        # outputssigmoid = np.concatenate(outputssigmoid, axis=0)
        # predictions = np.concatenate(predictions, axis=0)
        # labels = np.concatenate(labels, axis=0)



        # Print the results for each class
        #num_classes = len(precision)
        # for i in range(num_classes):
        #     print(f"Class {i + 1} - Precision: {precision[i]}, Recall: {recall[i]}, F1 Score: {f1_score[i]}")

        # ROC Curve and AUC
        fpr, tpr, thresholds = roc_curve(current_labels.cpu().numpy().ravel(), current_outputs.cpu().numpy().ravel())
        auc = roc_auc_score(current_labels.cpu().numpy().ravel(), current_outputs.cpu().numpy().ravel())

        # # Print the evaluation results
        # print("Confusion Matrix:")
        # print(confusion)
        # print("\nClassification Report:")
        # #print(report)
        # print("\nF1 Score:", f1_score)
        # print("Precision:", precision)
        # print("Recall:", recall)
        #print("AUC:", auc)

        # # Create a figure and axes for the F1 score plot
        # plt.figure(figsize=(10, 6))
        # plt.bar(range(1, num_classes + 1), f1_score)
        # plt.xlabel('Class Number')
        # plt.ylabel('F1 Score')
        # plt.title('F1 Score for each Class')
        # plt.xticks(range(1, num_classes + 1), range(1, num_classes + 1))  # Set class numbers as x-axis labels
        # plt.ylim(0, 1)  # Set the y-axis range (F1 score is between 0 and 1)
        # plt.grid(axis='y')
        # plt.tight_layout()

        # # Convert the plot to an image and log it to TensorBoard
        # f1_score_image = io.BytesIO()
        # plt.savefig(f1_score_image, format='png')
        # plt.close()
        # f1_score_image = Image.open(f1_score_image)

        # f1_score_image_np = np.array(f1_score_image)
        # f1_score_image_np = np.transpose(f1_score_image_np, (2,0,1))
        # f1_summary_writer.add_image(f'F1 Score {str(i)}', f1_score_image_np, global_step=epoch)

        # Plot ROC Curve
        plt.figure(figsize=(6, 10))
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'For oversampled')
        plt.legend(loc="lower right")


        
        # Convert the plot to an image and log it to TensorBoard
        ROC_image = io.BytesIO()
        plt.savefig(ROC_image, format='png')
        plt.close()
        ROC_image.seek(0)
        figs_oversampled.append(ROC_image)
    
        # # Convert the plot to an image and log it to TensorBoard
        # ROC_image = io.BytesIO()
        # plt.savefig(ROC_image, format='png')
        # plt.close()
        # ROC_image = Image.open(ROC_image)

        # # Convert the image to a numpy array and log it to TensorBoard using tf.summary.image
        # ROC_image_np = np.array(ROC_image)
        # ROC_image_np = np.transpose(ROC_image_np, (2,0,1))
        # ROC_summary_writer.add_image(f'Multi-Label ROC  {str(i)}', ROC_image_np) #''', global_step=epoch'''


    ''' 
    # Plot and save the train and validation line graphs
        plt.figure(figsize=(10, 7))
        plt.plot(train_loss, color='orange', label='train loss')
        plt.plot(valid_loss, color='red', label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_plot.png')
        plt.show()
    '''



model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Code/Models based on f1/model, good mesophagus highest f1 0.5077625513076782 06-08.pth'))
batch_size=64
class_weights = torch.tensor([1, 2.6, 3.7, 3.7, 4.0, 5.2, 10.4, 15.3, 18.1, 18.5]).to(device)

model.to(device)  # Move the model to the GPU

val_csv = pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_test.csv')
val_csv = dropzerolabels(val_csv)

# Evaluation dataset with test=True
eval_data = ImageDataset(val_csv, train=False, test=False)
eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)

predictions = []
labels = []
outputssigmoid = []
number_of_classes=10

STEP=0.01 #how often the threshold is evaluated per class

#czyli dla klasy fora, szukac threshold, albo wektor albo 10 petli for
with torch.no_grad():
    for data in tqdm(eval_loader):
        images, true_labels = data['image'].to(device), data['label'].to(device)
        outputs = model(images)
        outputs=torch.sigmoid(outputs * class_weights)

        outputssigmoid.append(outputs)
        labels.append(true_labels)

    outputssigmoid = torch.cat(outputssigmoid, dim=0)
    labels = torch.cat(labels, dim=0)
        
    for i in range(number_of_classes):
        print("Class:",i)
        matrix_log_dir = f'/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Temporary/Logs Threshold eval/Matrix Class{str(i)}'
        f1_log_dir = f'/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Temporary/Logs Threshold eval/f1 Class{str(i)}'
        ROC_log_dir = f'/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Temporary/Logs Threshold eval/ROC Class{str(i)}'
            
        matrix_summary_writer = tb.SummaryWriter(log_dir=matrix_log_dir)
        f1_summary_writer = tb.SummaryWriter(log_dir=f1_log_dir)
        ROC_summary_writer = tb.SummaryWriter(log_dir=ROC_log_dir)
        
        current_outputs = outputssigmoid[:, i]
        current_labels = labels[:, i]

        thresholds=np.arange(0,1,STEP)
        f1_scores_graph=[]

        for j in thresholds:
            #print("Threshold:",j)
            outputs_thresholded = current_outputs > j
            # Calculate Precision, Recall, and F1 Score for each class individually
            #precision, recall, f1_score, _ = precision_recall_fscore_support(current_labels, current_outputs, average=None)
            f1 = torchmetrics.classification.BinaryF1Score(multidim_average='global').to(device)
            f1 = f1(outputs_thresholded,current_labels)
            f1_summary_writer.add_scalar('F1', f1, global_step=j)
            #print(f1.cpu().numpy())
            f1_scores_graph.append(f1.cpu().numpy())
            Correct_label_count = sum(current_labels.cpu().numpy())

            # Confusion Matrix
            confusion = confusion_matrix(outputs_thresholded.cpu().numpy(), current_labels.cpu().numpy())
            confusion_image = torch.tensor(confusion).float().unsqueeze(0)  # Convert to a 1x2x2 tensor
            confusion_image = make_grid(confusion_image, nrow=1, normalize=True)  # Create a grid of 1x2 images
            # Log confusion matrix image to TensorBoard
            print(confusion)
            matrix_summary_writer.add_image(f'Multi-Label Confusion Matrix Class {str(i)}', confusion_image, global_step=j)
        
        plt.plot(thresholds,f1_scores_graph)
        plt.title(f"Positive cases:{Correct_label_count}")
        plt.show()


        # ROC Curve and AUC
        fpr, tpr, thresholds = roc_curve(current_labels.cpu().numpy().ravel(), current_outputs.cpu().numpy().ravel())
        auc = roc_auc_score(current_labels.cpu().numpy().ravel(), current_outputs.cpu().numpy().ravel())


        # Plot ROC Curve
        plt.figure(figsize=(6, 10))
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'For weights')
        plt.legend(loc="lower right")

        # Convert the plot to an image and log it to TensorBoard
        ROC_image_w = io.BytesIO()
        plt.savefig(ROC_image_w, format='png')
        plt.close()
        ROC_image_w.seek(0)
        figs_weighted.append(ROC_image_w)


# Define the number of classes
number_of_classes = 10

# Create subplots for each class
for i in range(number_of_classes):
    plt.figure(figsize=(12, 8))
    plt.suptitle(f'Class {i}', fontsize=16)

    # Add the ROC curve plot for weighted model
    plt.subplot(1, 2, 1)
    weighted_ROC_image = figs_weighted[i]
    weighted_ROC_image.seek(0)
    plt.imshow(plt.imread(weighted_ROC_image))
    plt.axis('off')
    plt.title('Weighted')

    # Add the ROC curve plot for oversampled model
    plt.subplot(1, 2, 2)
    oversampled_ROC_image = figs_oversampled[i]
    oversampled_ROC_image.seek(0)
    plt.imshow(plt.imread(oversampled_ROC_image))
    plt.axis('off')
    plt.title('Oversampled')

    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplot layout
    plt.show()