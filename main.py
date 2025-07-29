import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from torchsummary import summary
#Local imports
from utils import create_y_labels, plot_classification_metrics, compute_segmentation_weights
from train_classification import classification_pipeline, resnet_transfer_pipeline
from train_segmentation import PSP_like_segmentation_pipeline, transfer_segmentation_pipeline
from networks.segmentation_network import SegmentationNet
from train_vqa import vqa_pipeline

if __name__ == "__main__":
    #Creation of the y_labels for classification
    #create_y_labels('Dataset\\FloodNet-Supervised_v1.0\\train\\train-label-img', "train")
    #create_y_labels('Dataset\\FloodNet-Supervised_v1.0\\val\\val-label-img', "val")
    #create_y_labels('Dataset\\FloodNet-Supervised_v1.0\\test\\test-label-img', "test")
    #classification_pipeline()
    #plot_classification_metrics()
    #resnet_transfer_pipeline() 
    #segmentation_pipeline()
    #PSP_like_segmentation_pipeline()
    #transfer_segmentation_pipeline()
    vqa_pipeline()