import torch
from tqdm import tqdm
# training function
def train(model, dataloader, optimizer, criterion, train_data, device):
    print('Training')
    model.train()
    counter = 0
    train_running_loss = 0.0
    train_correct_preds = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        counter += 1
        data, target = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(outputs)
        #print(outputs)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        
        # calculate accuracy
        predicted_labels = (outputs > 0.5).float()  # convert probabilities to binary predictions
        
        train_correct_preds += (predicted_labels == target).sum().item()

        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        
    train_loss = train_running_loss / counter
    train_accuracy = train_correct_preds / (counter * dataloader.batch_size)

    return train_loss, train_accuracy

def validate(model, dataloader, criterion, val_data, device):
    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    val_correct_preds = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            counter += 1
            data, target = data['image'].to(device), data['label'].to(device)
            outputs = model(data)
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, target)
            val_running_loss += loss.item()

            # calculate accuracy
            predicted_labels = (outputs > 0.5).float()
            val_correct_preds += (predicted_labels == target).sum().item()
        
        val_loss = val_running_loss / counter
        val_accuracy = val_correct_preds / (counter * dataloader.batch_size)
        return val_loss, val_accuracy