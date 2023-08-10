from torchvision import models
import torch.nn as nn

def model(pretrained, requires_grad):
    model = models.resnet50(progress=True, pretrained=pretrained)
    # to freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layer learnable
    # we have 10 classes in total
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

# model = models.resnet18(progress=True, pretrained=pretrained)
#     # to freeze the hidden layers
#     if requires_grad == False:
#         for param in model.parameters():
#             param.requires_grad = False
#     # to train the hidden layers
#     elif requires_grad == True:
#         for param in model.parameters():
#             param.requires_grad = True
#     # make the classification layer learnable
#     # we have 10 classes in total
#     model.fc = nn.Linear(model.fc.in_features, 10)
#     return model