from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.models as models
import torchvision.transforms as transforms
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
import pandas as pd
import PIL

# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--use-cuda', action='store_true', default=True,
#                         help='Use NVIDIA GPU acceleration')
#     parser.add_argument(
#         '--image-path',
#         type=str,
#         default='/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/Images/00163034.dcm.png',
#         help='Input image path')
#     parser.add_argument('--aug_smooth', action='store_true',
#                         help='Apply test time augmentation to smooth the CAM')
#     parser.add_argument(
#         '--eigen_smooth',
#         action='store_true',
#         help='Reduce noise by taking the first principle componenet'
#         'of cam_weights*activations')

#     parser.add_argument(
#         '--method',
#         type=str,
#         default='gradcam',
#         help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

#     args = parser.parse_args()
#     args.use_cuda = args.use_cuda and torch.cuda.is_available()
#     if args.use_cuda:
#         print('Using GPU for acceleration')
#     else:
#         print('Using CPU for computation')

#     return args


# def reshape_transform(tensor, height=14, width=14):
#     result = tensor[:, 1:, :].reshape(tensor.size(0),
#                                       height, width, tensor.size(2))

#     # Bring the channels to the first dimension,
#     # like in CNNs.
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result

# args = get_args()
# methods = \
#     {"gradcam": GradCAM,
#         "scorecam": ScoreCAM,
#         "gradcam++": GradCAMPlusPlus,
#         "ablationcam": AblationCAM,
#         "xgradcam": XGradCAM,
#         "eigencam": EigenCAM,
#         "eigengradcam": EigenGradCAM,
#         "layercam": LayerCAM,
#         "fullgrad": FullGrad}

# if args.method not in list(methods.keys()):
#     raise Exception(f"method should be one of {list(methods.keys())}")




# from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
# from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange



# # State of the art metric: Remove and Debias
# from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst, ROADLeastRelevantFirst

# from pytorch_grad_cam.metrics.road import ROADMostRelevantFirstAverage, ROADLeastRelevantFirstAverage, ROADCombined




# import tensorflow as tf
# from tensorflow import keras
# from PIL import Image
# from IPython.display import display
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import numpy as np
# import cv2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.transforms.functional import to_pil_image
# from matplotlib import colormaps
# import numpy as np
# import pandas as pd
# import PIL



# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 10)
# model.load_state_dict(torch.load('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Code/final model, oversampled 08-08.pth', map_location=torch.device('cpu')))
# model.eval()  # Set the model to evaluation mode

# img_size = (224,224)
# thresholds=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
# target_class=0

# for threshold in thresholds:
#     #backward_hookk = model.layer4[1].conv2.register_full_backward_hook(backward_hook)
#     #forward_hookk = model.layer4[1].conv2.register_forward_hook(forward_hook)
    
#     if args.use_cuda:
#         model = model.cuda()

#     target_layers = model.layer4[1].conv2

#     if args.method not in methods:
#         raise Exception(f"Method {args.method} not implemented")

#     if args.method == "ablationcam":
#         cam = methods[args.method](model=model,
#                                     target_layers=target_layers,
#                                     use_cuda=args.use_cuda,
#                                     reshape_transform=reshape_transform,
#                                     ablation_layer=AblationLayerVit())
#     else:
#         cam = methods[args.method](model=model,
#                                     target_layers=target_layers,
#                                     use_cuda=args.use_cuda,
#                                     reshape_transform=reshape_transform)

#     # The local path to our target image
#     img_name='SO-0641-0001-0001.dcm'
#     img_path = f"/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/Images/{img_name}.png"


#     # Preprocess the image
#     preprocess = transforms.Compose([
#         transforms.Resize(img_size),
#         transforms.Grayscale(num_output_channels=3),
#         transforms.ToTensor()
#     ])

#     # Prepare image
#     image = Image.open(img_path)
#     img_tensor = preprocess(image)
#     input_tensor=img_tensor.unsqueeze(0)

#     #prepare the True labels
#     df_img=pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_test.csv')
#     df_img=df_img[df_img['FileName'].str.contains(f'{img_name}')]
#     true_labels=np.array(df_img.drop(['FileName', 'TAG'], axis=1))


#     output = model(input_tensor)
#     print(f'Class {target_class+1} guess is=', str(np.array(torch.sigmoid(output[0][target_class])>threshold)))
    
#     if int(np.array(torch.sigmoid(output[0][target_class])>threshold).flatten()) == true_labels[0][target_class] & true_labels[0][target_class]==1:
#         cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

#         targets = [ClassifierOutputTarget(281)]

#         # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
#         grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

#         grayscale_cam = grayscale_cam[0, :]
#         visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)

#         # Create the metric target, often the confidence drop in a score of some category
#         metric_target = ClassifierOutputSoftmaxTarget(281)
#         scores, batch_visualizations = CamMultImageConfidenceChange()(input_tensor, inverse_cams, targets, model, return_visualization=True)
#         visualization = deprocess_image(batch_visualizations[0, :])
#         # State of the art metric: Remove and Debias
#         cam_metric = ROADMostRelevantFirst(percentile=75)
#         scores, perturbation_visualizations = cam_metric(input_tensor, grayscale_cams, targets, model, return_visualization=True)
#         # You can also average accross different percentiles, and combine
#         # (LeastRelevantFirst - MostRelevantFirst) / 2
#         cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
#         scores = cam_metric(input_tensor, grayscale_cams, targets, model)
        
#     target_class+=1

import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default=f"/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/Images/IM-0140-0002.dcm.png",
        help='Input image path')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='eigengradcam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory to save the images')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise
    }

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Code/final model, oversampled 08-08.pth', map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    target_layers = [model.layer4[1].conv2]
    img_size = (224,224)
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    threeDprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3)
    ])


    rgb_imgg = Image.open(args.image_path)
    rgb_img=np.float32(threeDprocess(rgb_imgg)) / 255
    input_tensor = preprocess(rgb_imgg).unsqueeze(0)

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(range of classes 0-9)]
    targets = [ClassifierOutputTarget(6)]

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    #cam_algorithm = methods[args.method]
    for method_name, cam_algorithm in methods.items():
        with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=args.use_cuda) as cam:


            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            grayscale_cam = grayscale_cam[0, :]
            np.divide(grayscale_cam, np.max(grayscale_cam))
            
            mask = grayscale_cam > 0.7

            # Apply the mask to the array to filter out smaller values
            grayscale_camm = np.where(mask, grayscale_cam, 0)


            # Define the Gaussian blur parameters
            kernel_size = (15, 15)  # Size of the Gaussian kernel
            sigma_x = 5             # Standard deviation along X-axis (automatically calculated based on kernel size)
            # Apply Gaussian blur
            grayscale_camm = cv2.GaussianBlur(grayscale_camm, kernel_size, sigma_x)


            #IMPORTANT  - in order to show good images for just numbers in name, image must be cubed
            cam_image = show_cam_on_image(rgb_img**1, grayscale_camm**2.5, use_rgb=False, colormap = cv2.COLORMAP_TURBO, image_weight = 0.75)
            heatmap=show_cam_on_image(rgb_img**1, grayscale_camm**2.5, use_rgb=False, colormap = cv2.COLORMAP_TURBO, image_weight = 0)
            #cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        gb = gb_model(input_tensor, target_category=None)

        cam_mask = cv2.merge([grayscale_camm, grayscale_camm, grayscale_camm])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        os.makedirs(args.output_dir, exist_ok=True)

        cam_output_path = os.path.join(args.output_dir, f'{method_name}_cam.jpg')
        gb_output_path = os.path.join(args.output_dir, f'{method_name}_gb.jpg')
        cam_gb_output_path = os.path.join(args.output_dir, f'{method_name}_heatmap.jpg')
        cv2.imwrite(cam_output_path, cam_image)
        #cv2.imwrite(gb_output_path, gb)
        cv2.imwrite(cam_gb_output_path, heatmap)



#target_layers = model.layer4[1].con2


# img_size = (224,224)
# # The local path to our target image
# img_name='SO-0641-0001-0001.dcm'
# img_path = f"/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/Images/{img_name}.png"

# # Preprocess the image
# preprocess = transforms.Compose([
#     transforms.Resize(img_size),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor()
# ])

# # Prepare image
# image = Image.open(img_path)
# img_tensor = preprocess(image)
# input_tensor=img_tensor.unsqueeze(0)
# thresholds=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
# target_class=0

# for threshold in thresholds:

   
#     # The local path to our target image
#     img_name='SO-0641-0001-0001.dcm'
#     img_path = f"/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/Images/{img_name}.png"


#     # Preprocess the image
#     preprocess = transforms.Compose([
#         transforms.Resize(img_size),
#         transforms.Grayscale(num_output_channels=3),
#         transforms.ToTensor()
#     ])

#     # Prepare image
#     image = Image.open(img_path)
#     img_tensor = preprocess(image)
#     input_tensor=img_tensor.unsqueeze(0)

#     #prepare the True labels
#     df_img=pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_test.csv')
#     df_img=df_img[df_img['FileName'].str.contains(f'{img_name}')]
#     true_labels=np.array(df_img.drop(['FileName', 'TAG'], axis=1))


#     output = model(input_tensor)
#     print(torch.sigmoid(output[0][target_class]))
#     print(f'Class {target_class+1} guess is=', str(np.array(torch.sigmoid(output[0][target_class])>threshold)))
    
#     if int(np.array(torch.sigmoid(output[0][target_class])>threshold).flatten()) == true_labels[0][target_class] & true_labels[0][target_class]==1:

#         model.cuda()
#         input_tensor = input_tensor.cuda()
#         np.random.seed(42)
#         benchmark(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False)
    
#     target_class+=1