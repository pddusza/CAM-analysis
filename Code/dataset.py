import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, resnet18
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import multilabel_confusion_matrix, classification_report, precision_recall_fscore_support, roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from DropZeroValueLabels import dropzerolabels
import torch.utils.tensorboard
import tensorflow as tf
import datetime
from DatasetPreparation import DatasetPreparation


csv = pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP.csv')

train_csv, remaining = train_test_split(
    csv, train_size=0.70, shuffle=True, random_state=42
)
val_csv, test_csv = train_test_split(
    remaining, train_size=0.3, shuffle=True, random_state=42
)


train_csv.to_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_train.csv', index=False)

val_csv.to_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_val.csv', index=False)

test_csv.to_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP_test.csv', index=False)
