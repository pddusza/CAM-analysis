import models
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from engine import train, validate
from dataset import ImageDataset
from torch.utils.data import DataLoader
matplotlib.style.use('ggplot')
# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#intialize the model
model = models.model(pretrained=True, requires_grad=False).to(device)
# learning parameters
lr = 0.0001
epochs = 10 
batch_size = 32
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()
# read the training csv file
train_csv = pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP.csv')
# train dataset
train_data = ImageDataset(
    train_csv, train=True, test=False
)
# validation dataset
valid_data = ImageDataset(
    train_csv, train=False, test=False
)
# train data loader
train_loader = DataLoader(
    train_data, 
    batch_size=batch_size,
    shuffle=True
)
# validation data loader
valid_loader = DataLoader(
    valid_data, 
    batch_size=batch_size,
    shuffle=False
)

# start the training and validation
train_loss = []
valid_loss = []
train_accuracy = []
valid_accuracy = []

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_accuracy = train(model, train_loader, optimizer, criterion, train_data, device)
    valid_epoch_loss, valid_epoch_accuracy = validate(model, valid_loader, criterion, valid_data, device)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    valid_accuracy.append(valid_epoch_accuracy)
    print(f"Train Loss: {train_epoch_loss:.4f}, \tTrain Accuracy: {train_epoch_accuracy:.4f}")
    print(f"Validation Loss: {valid_epoch_loss:.4f}, \tValidation Accuracy: {valid_epoch_accuracy:.4f}")


    # save the trained model to disk
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, '/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Results/test dp/model.pth')
# plot and save the train and validation line graphs
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Results/test dp')
plt.show()