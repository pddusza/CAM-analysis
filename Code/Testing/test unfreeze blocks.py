import torch
import torchvision.models as models
import numpy as np
import copy

# Load the ResNet-18 model
model = models.resnet18(pretrained=True)

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
        
    print(block_name,b)


#         # Verify that the layers have been unfrozen accordingly
#         for name, param in model.named_parameters():
#             if param.requires_grad==True:
#                 print(f'{name}: requires_grad={param.requires_grad}')
#         print()
# #print('\n\n',modela,'\n\n')

all_requires_grad_true = all(param.requires_grad for param in model_list[17].parameters())
print(all_requires_grad_true) 
print('\n\n\n\n')

# for model in model_list:
#     for name, param in model.named_parameters():
#         if param.requires_grad==True:
#             print(f'{name}: requires_grad={param.requires_grad}')
#     print()