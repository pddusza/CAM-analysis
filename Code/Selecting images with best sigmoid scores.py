import torch.utils.tensorboard as tb
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from DropZeroValueLabels import dropzerolabels
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import io
from PIL import Image
import seaborn as sns
from torchvision.utils import make_grid


class ImageDataset(Dataset):
    def __init__(self, csv, train, test):
        self.csv = csv
        self.train = train
        self.test = test

        if self.test == True and self.train == False:
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
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        image = self.transform(image)
        targets = self.labels[index]
        filename = self.image_names[index]

        return {
            'image':image,
            'label': torch.tensor(targets, dtype=torch.float32),
            'name': filename}



figs_weighted=[]
figs_oversampled=[]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Code/final model, oversampled 08-08.pth'))
batch_size=64

model.to(device)  # Move the model to the GPU

test_csv = pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_test.csv')
test_csv = dropzerolabels(test_csv)

# Evaluation dataset with test=True
test_data = ImageDataset(test_csv, train=False, test=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

predictions = []
labels = []
outputssigmoid = []
number_of_classes=10

class_filename_dict = {i: {} for i in range(number_of_classes)}

STEP=0.01 #how often the threshold is evaluated per class

#czyli dla klasy fora, szukac threshold, albo wektor albo 10 petli for
classwise_results = {}  # Dictionary to store class-wise results

with torch.no_grad():
    for data in tqdm(test_loader):
        images, true_labels, FileName = data['image'].to(device), data['label'].to(device), data['name']
        outputs = model(images)
        outputs = torch.sigmoid(outputs)

        for i in range(number_of_classes):
            current_outputs = outputs[:, i]
            current_labels = true_labels[:, i]
            class_name = f"Class_{i}"  # You can adjust this to match your class naming convention
            
            # Initialize an empty list for the class if it doesn't exist
            if class_name not in classwise_results:
                classwise_results[class_name] = []
            
            # Append filename and sigmoid value to the class-wise list
            for filename, sigmoid_value, label in zip(FileName, current_outputs, current_labels):
                if label == 1:  # Only consider filenames with positive labels
                    classwise_results[class_name].append((filename, sigmoid_value.item()))

# Sort each class's results based on sigmoid values in descending order
for class_name in classwise_results:
    classwise_results[class_name] = sorted(classwise_results[class_name], key=lambda x: x[1], reverse=True)

# Print or process the sorted class-wise results
for class_name, results in classwise_results.items():
    print("Class:", class_name)
    for filename, sigmoid_value in results:
        print("Filename:", filename, "Sigmoid:", sigmoid_value)

