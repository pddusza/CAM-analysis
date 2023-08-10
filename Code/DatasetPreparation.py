import torch
from torch.utils.data import Dataset
import kornia
import kornia.augmentation as Kaug
import SimpleITK as sitk
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import sys
import numpy as np
import cv2
from Multilabel_Binary_DataFrame import Multilabel_Binary_DataFrame
from PNGtextremover import DeletePNGfromFileName

def DatasetPreparation(desired_class,desired_specie):
    class DataFrameDataset(Dataset):
        def __init__(self, dataframe, transform=None, augmentation_names=None):
            self.dataframe = dataframe
            self.transform = transform
            self.augmentation_names = augmentation_names

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            row = self.dataframe.iloc[idx]
            image_path = row['Path']
            image_name = row['FileName']
            label = row['TAG']

            #Image loading using SimpleITK
            image = sitk.ReadImage(image_path)
            image_data = sitk.GetArrayFromImage(image)
            image_data = image_data.astype(float)  # Convert to float32
            image_tensor = torch.from_numpy(image_data)

            #Image to torch.Tensor convertion
            image_tensor = image_tensor.unsqueeze(0).float()

            augmented_images = []
            augmented_labels = []
            augmented_filenames = []

            #Kornia transformations for image augmentation
            if self.transform is not None:
                for i, transform in enumerate(self.transform):
                    resized_image = Kaug.Resize((224, 224))(image_tensor)
                    augmented_image = transform(resized_image)
                    augmented_images.append(augmented_image.squeeze())
                    augmented_labels.append(f'{label}')
                    #augmented_labels.append(f'{label} + {self.augmentation_names[i]}')  #Used only for printing the augmented names in labels
                    augmented_filenames.append(f'{image_name}.png')

            return augmented_images, augmented_labels, augmented_filenames


    df_all = pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Results/DataFrame_multiclass_excluded_correctspelling(3).csv')

    #Excluding all the other scan positions
    mask = df_all['Projection'] == 'LL'
    df_class = df_all[mask]

    mask = df_class['specie'] == 'cane'
    df = df_class[mask]

    #Kornia augmentation transforms
    transforms = [
        Kaug.RandomHorizontalFlip(p=0), #basical original (just resized) since probabiloty = 0
        # Kaug.RandomHorizontalFlip(p=1),
        # Kaug.RandomVerticalFlip(p=1),
        # Kaug.RandomRotation(degrees=30,p=1),
        # Kaug.RandomPerspective(distortion_scale=0.1,p=1),
        #Kaug.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) #nie dziala na czarnobialym
    ]

    aug_nam = [
        'original',
        'horizontal_flip',
        'vertical_flip',
        'rotation',
        'perspective',
        #'color_jitter' #nie dziala na czarnobialym
    ]
    augmented_data = []

    dataset = DataFrameDataset(df, transform=transforms)
    #dataset = DataFrameDataset(df, transform=transforms, augmentation_names=aug_nam)   #Code used for veryfying augmentations when printing images
    print(dataset)
    print(len(dataset))


    output_dir = '/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/Images'  # Specify the directory to save the augmented images
    os.makedirs(output_dir, exist_ok=True)


    for i in range(len(dataset)):
        images, labels, filenames = dataset[i]
        for j in range(len(images)):
            image = images[j]
            label = labels[j]

            image = image.transpose(-2,1).transpose(0, 1)  # Convert tensor to image format (HWC)
            output_path = os.path.join(output_dir, f'{filenames[j]}')
            #plt.imshow(image, cmap=plt.cm.bone)
            #plt.title(f"Label: {label}")
            #plt.axis('off')
            image = image.numpy()  # Convert tensor to NumPy array

            #Brightness normalization
            image=np.divide(np.subtract(image, np.min(image)), np.subtract(np.max(image),np.min(image)))
            image=np.multiply(image, 255)
            norm_brightness=1/(np.max(image)/255)
            
            image=np.multiply(image, norm_brightness)
            
            second_max_value = np.partition(image, -2)[-2]
            image = np.where(image == np.max(image), second_max_value, image)
            
            image = image.astype('int16')  # Convert the array to unsigned 8-bit integers (0-255)
            
            np.set_printoptions(threshold=sys.maxsize)
            #print(np.max(image), np.min(image), image)
        
            cv2.imwrite(output_path, image)         #pil_image = Image.fromarray(image)  # Convert the NumPy array to a PIL image               #pil_image = pil_image.resize((224, 224), Image.BICUBIC)  # Resize the image to 224x224           #pil_image.save(output_path)
            #plt.show()
        
            augmented_data.append({
                'FileName': filenames[j],
                #'Path': df.iloc[i]['Path'],
                'TAG': labels[j],
                # Add other columns from the original DataFrame if needed
            }) #Making new DataFrame with augmented images    
        
        if(i%100==0):
            print(i,"/", len(dataset))


    new_df = pd.DataFrame(augmented_data)

    # Save the new DataFrame to a CSV file
    #new_df.to_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Results/DataFrame_test1.csv', index=False)

    df_Binary=Multilabel_Binary_DataFrame(new_df)

    csv = DeletePNGfromFileName(df_Binary)

    train_csv, remaining = train_test_split(
        csv, train_size=0.70, shuffle=True, random_state=42
    )
    val_csv, test_csv = train_test_split(
        remaining, train_size=0.3, shuffle=True, random_state=42
    )

    train_csv.to_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_train.csv', index=False)

    val_csv.to_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_val.csv', index=False)

    test_csv.to_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_test.csv', index=False)
