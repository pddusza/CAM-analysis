import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch.nn as nn
import torchvision.models as models

# # Define the GradCAM class
# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradient = None

#     def backward_hook(self, module, grad_input, grad_output):
#         self.gradient = grad_output[0]

#     def generate(self, input_image, target_class):
#         # Register the backward hook on the target layer
#         hook = self.target_layer.register_backward_hook(self.backward_hook)

#         class_index = target_class.index(target_class)

#         # Perform forward pass
#         model_output = model(input_image)
#         target_output = model_output[0, class_index]

#         # Zero the gradients
#         self.model.zero_grad()
#         target_output.backward()

#         # Remove the hook
#         hook.remove()

#         # Get the gradients and features
#         gradient = self.gradient[0].mean(dim=(1, 2), keepdim=True)  # Adjusted for 224x224 images
#         features = F.relu(self.target_layer(input_image))  # Adjusted for 224x224 images

#         # Calculate GradCAM heatmap
#         grad_cam = (features * gradient).sum(dim=1, keepdim=True)
#         grad_cam = F.relu(grad_cam)

#         return grad_cam

# # Load your CSV file containing filenames and labels
# csv_path = '/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_test.csv'
# data_df = pd.read_csv(csv_path)

# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 10)
# model.load_state_dict(torch.load('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Code/final model, oversampled 08-08.pth'))
# model.eval()  # Set the model to evaluation mode


# # Define the target layer for GradCAM
# target_layer = model.layer3[0].conv2  # Modify this based on your model architecture

# # Iterate over the dataset and generate GradCAM visualizations
# for index, row in data_df.iterrows():
#     image_name = row['FileName']  # Modify this based on your CSV structure
#     image = Image.open(f"/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/Images/{image_name}.png")
#     # Convert single-channel (grayscale) image to three-channel (RGB)
#     image_rgb = transforms.Grayscale(num_output_channels=3)(image)
    
#     # Preprocess the image
#     preprocess = transforms.Compose([
#         transforms.Resize((256,256)),
#         transforms.ToTensor()
#     ])
#     input_image = preprocess(image_rgb).unsqueeze(0)  # Add batch dimension
    
#     # Get the target labels
#     target_labels = row # Label columns start from the third column
#     print(target_labels)
#     for label_name, label_value in target_labels.items():
#         print(label_value)
#         print(label_name)
#         if label_value == 1:  # GradCAM for active labels only
#             grad_cam = GradCAM(model, target_layer)
#             heatmap = grad_cam.generate(input_image, label_name) 

#             # Normalize the heatmap
#             heatmap_normalized = heatmap / torch.max(heatmap)
            
#             # Overlay heatmap on the original image
#             heatmap_overlay = transforms.ToPILImage()(heatmap_normalized)
#             result_image = Image.blend(image.convert('RGBA'), heatmap_overlay.convert('RGBA'), alpha=0.5)
            
#             # Save or display the result_image
#             result_image.show()




'''#ten tutaj dziala ale jest prosty gradient zrobiony

# import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Code/final model, oversampled 08-08.pth'))
model.eval()  # Set the model to evaluation mode

img_size = (224,224)

# The local path to our target image
img_path = "/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/Images/IMA-0388-0001.dcm.png"


# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(img_size),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])


# Prepare image
image = Image.open(img_path)
input_tensor = preprocess(image)
input_tensor = input_tensor.unsqueeze(0)

# Calculate gradients for each class
num_classes = len(model.fc.weight)
for target_class in range(num_classes):
    input_tensor_clone = input_tensor.clone().requires_grad_(True)

    output = model(input_tensor_clone)
    loss = output[0, target_class]

    model.zero_grad()
    loss.backward()

    gradients = input_tensor_clone.grad.squeeze().cpu().numpy()
    gradients = gradients.transpose(1, 2, 0)

    # Visualize
    plt.figure(figsize=(10, 8))
    plt.imshow(gradients, cmap='viridis', alpha=0.5)
    plt.imshow(image, alpha=0.7, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title(f'Gradient Map for Class {target_class}')
    plt.show()
'''




# #input_tensor = input_tensor.permute(2, 3,1, 0)  # Transpose the tensor

# # Calculate gradients and generate Grad-CAM heatmap for each class
# num_classes = 10  # the number of output classes
# for target_class in range(num_classes):
#     input_tensor_clone = input_tensor.clone().requires_grad_(True)
    
#     output = model(input_tensor_clone)
#     target_output = output[0, target_class]

#     model.zero_grad()
#     target_output.backward()

#     gradients = input_tensor_clone.grad.squeeze().cpu().numpy()
#     pooled_gradients = np.mean(gradients, axis=(1, 2))
    
#     activations = model.layer1[0].conv2(input_tensor_clone)
#     activations = activations.cpu().detach().numpy()[0]

#     weighted_activations = np.mean(activations * pooled_gradients[:, np.newaxis, np.newaxis], axis=0)
#     grad_cam = np.maximum(weighted_activations, 0)

#     # Normalize the heatmap
#     heatmap = grad_cam / np.max(grad_cam)

#     # Resize the heatmap to match the original image size
#     heatmap = cv2.resize(heatmap, (image.width, image.height))

#     # Apply heatmap on the original image
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     superimposed_img = cv2.addWeighted(np.array(image), 0.7, heatmap, 0.3, 0)

#     # Visualize Grad-CAM heatmap
#     plt.figure(figsize=(10, 8))
#     plt.imshow(superimposed_img)
#     plt.axis('off')
#     plt.title(f'Grad-CAM Heat Map for Class {target_class}')
#     plt.show()





import tensorflow as tf
from tensorflow import keras
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps
import numpy as np
import PIL



# hyperparameters
nc = 3 # number of channels
nf = 64 # number of features to begin with
dropout = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# setup a resnet block and its forward function
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self(x)
        out = F.relu(out)
        return out

# setup the final model structure
class CustomResNet18(nn.Module):
    def __init__(self, nc=nc, nf=nf, dropout=dropout):
        super(CustomResNet18, self).__init__()

        self.layer = nn.Sequential(
            ResNetBlock(nc,   nf,    stride=2), # (B, C, H, W) -> (B, NF, H/2, W/2), i.e., (64,64,128,128)
            ResNetBlock(nf,   nf*2,  stride=2), # (64,128,64,64)
            ResNetBlock(nf*2, nf*4,  stride=2), # (64,256,32,32)
            ResNetBlock(nf*4, nf*8,  stride=2), # (64,512,16,16)
            ResNetBlock(nf*8, nf*16, stride=2), # (64,1024,8,8)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(nf*16, 1, 8, 1, 0, bias=False),
            nn.Dropout(p=dropout),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.layer(input.to(device))
        output = self.classifier(output)
        return output




import tensorflow as tf
from tensorflow import keras
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps
import numpy as np
import PIL


# defines two global scope variables to store our gradients and activations
gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
  global gradients # refers to the variable in the global scope
  print('Backward hook running...')
  gradients = grad_output
  # In this case, we expect it to be torch.Size([batch size, 512, 7, 7])
  print(f'Gradients size: {gradients[0].size()}') 
  # We need the 0 index because the tensor containing the gradients comes
  # inside a one element tuple.

def forward_hook(module, args, output):
  global activations # refers to the variable in the global scope
  print('Forward hook running...')
  activations = output
  # In this case, we expect it to be torch.Size([batch size, 512, 7, 7])
  print(f'Activations size: {activations.size()}')

#model=CustomResNet18()
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Code/final model, oversampled 08-08.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

img_size = (224,224)
thresholds=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
target_class=0

for threshold in thresholds:
    backward_hookk = model.layer4[1].conv2.register_full_backward_hook(backward_hook)
    forward_hookk = model.layer4[1].conv2.register_forward_hook(forward_hook)

    # The local path to our target image
    img_name='SO-0641-0001-0001.dcm'
    img_path = f"/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/Images/{img_name}.png"


    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    # Prepare image
    image = Image.open(img_path)
    img_tensor = preprocess(image)
    input_tensor=img_tensor.unsqueeze(0)

    #prepare the True labels
    df_img=pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_test.csv')
    df_img=df_img[df_img['FileName'].str.contains(f'{img_name}')]
    true_labels=np.array(df_img.drop(['FileName', 'TAG'], axis=1))



    output = model(input_tensor)
    print(f'Class {target_class+1} guess is=', str(np.array(torch.sigmoid(output[0][target_class])>threshold)))
    
    if int(np.array(torch.sigmoid(output[0][target_class])>threshold).flatten()) == true_labels[0][target_class] & true_labels[0][target_class]==1:
        loss = output[0, target_class]
        loss.backward()
        
        pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])

        for i in range(activations.size()[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        heatmap = F.relu(heatmap)

        # normalize the heatmap
        heatmap -= torch.min(heatmap)
        heatmap /= torch.max(heatmap)

        
        # draw the heatmap
        plt.matshow(heatmap.detach())

        # Create a figure and plot the first image
        fig, ax = plt.subplots()
        ax.axis('off') # removes the axis markers

        # First plot the original image
        ax.imshow(to_pil_image(img_tensor, mode='RGB'))

        # Resize the heatmap to the same size as the input image and defines
        # a resample algorithm for increasing image resolution
        # we need heatmap.detach() because it can't be converted to numpy array while
        # requiring gradients
        overlay = to_pil_image(heatmap.detach(), mode='F').resize((224,224), resample=PIL.Image.BICUBIC)

        # Apply any colormap you want
        cmap = colormaps['jet']
        overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

        # Plot the heatmap on the same axes, 
        # but with alpha < 1 (this defines the transparency of the heatmap)
        ax.imshow(overlay, alpha=0.4, interpolation='nearest')

        # Show the plot
        plt.title(f"Class {target_class+1}")
        plt.show()

    forward_hookk.remove()
    backward_hookk.remove()
    target_class+=1

#model(input_tensor.unsqueeze(0)).backward(torch.Tensor([1, 1]))
