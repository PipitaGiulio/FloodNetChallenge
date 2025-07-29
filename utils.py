import numpy as np
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
import json
import string

from sklearn.metrics import confusion_matrix 
import cv2
from torch.nn.functional import one_hot
from torchmetrics.segmentation import DiceScore
#local imports
from networks.segmentation_network import PSP_SegmentationNet, Resnet_Transfer_SegNet
from networks.classification_network import ClassificationNetwork 
from networks.vqa_network import VQA_Net
from datasets.segmentation_dataset import SegmentationDataset

#file containing miscellaneous utils function used throughout the models development

class_to_color = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (180, 120, 120),
    3: (160, 150, 20),
    4: (140, 140, 140),
    5: (61, 230, 250),
    6: (0, 82, 255),
    7: (255, 0, 245),
    8: (255, 235, 0),
    9: (4, 250, 7)
}

color_to_class = {v: k for k, v in class_to_color.items()}


def create_y_labels(path_label_img, type):
    #This function is only used once to create the y tensor of labels used for the classification task
    #If pixel value = 1 or = 3 it will count as a 'flooded' pixel, therefore if in an image >30%
    #Of the pixel belongs to class 1 or 3, the scene represented in the image will be one of a flood
    y = []
    save_path =f'./Dataset/y_classification/y_{type}.npy'
    n_files = len(os.listdir(path_label_img))
    for i, file_name in enumerate(os.listdir(path_label_img)):
        img_path = os.path.join(path_label_img, file_name)
        img = Image.open(img_path)           
        img = np.array(img)
        tot_pixels = np.prod(img.shape)
        flooded_pixels = np.isin(img, [1, 3]).sum()
        flooded_percent = flooded_pixels/tot_pixels
        y.append(1) if flooded_percent > 0.3 else y.append(0)
        if i % 25 == 0:
            print(f"Processed {i}/{n_files} {type} images")
    y = np.array(y)
    np.save(save_path, y)


def compute_positional_weight(y_label):
    #compute y_labels weight for classification
    flooded = (y_label == 1).sum()
    non_flooded = (y_label == 0).sum()
    #flooded has class 1, so positional weight only accounts for them
    positional_weight = non_flooded/flooded
    #convert to tensor to be able to use it with pytorch tools (such as BCEWithLogitsLoss)
    return torch.tensor(positional_weight)

def compute_label_weight(y_label):
    #for resnet
    flooded = (y_label == 1).sum().item()
    non_flooded = (y_label == 0).sum().item()
    weight_flooded = len(y_label)/flooded
    weight_non_flooded = len(y_label)/non_flooded
    return torch.tensor([weight_non_flooded, weight_flooded], dtype = torch.float32).to('cuda')

def compute_segmentation_weights(y_path):
    #compute class weight for segmentation
    total_count = np.zeros(10, dtype=np.int64)
    for file in os.listdir(y_path):
        img = np.array(Image.open(os.path.join(y_path, file))).flatten()
        total_count += np.bincount(img, minlength=10)
    total_pixels = total_count.sum()
    weights = total_pixels/(total_count + 0.00000000000001)
    weights = weights/weights.mean()
    weights = np.clip(weights, a_min=0.5, a_max=5.0)
    print("Class counts:", total_count)
    print(weights)
    return torch.tensor(weights, dtype=torch.float32).to('cuda')


###utils function to plot loss, IoU, accuracy and confusion matrices of various models
def plot_classification_metrics():
    last_model = torch.load('.\\Models\\last_classification_model.pth', weights_only=False)
    accuracy_training = last_model['train_acc']
    loss_training = last_model['train_loss']
    accuracy_validation = last_model['val_accuracy']
    loss_validation = last_model['val_loss']
    f1 = plt.figure(1)
    plt.plot(accuracy_training, label = "Training Accuracy")
    plt.plot(accuracy_validation, label = "Validation Accuracy")
    plt.title("Accuracy Over Epochs for Classification Model Created From Scratch")
    plt.grid()
    plt.legend()
    f2 = plt.figure(2)
    plt.plot(loss_training, label = "Training Loss")
    plt.plot(loss_validation, label = "Validation Loss")
    plt.title("Loss Over Epochs for Classification Model Created From Scratch")
    plt.grid()
    plt.legend()
    plt.show()


def plot_segmentation_metrics():
    
    last_model = torch.load('.\\Models\\2TPSP_WDice_last_segmentation_model.pth', weights_only=False)
    
    accuracy_training = last_model['train_accuracy']
    loss_training = last_model['train_loss']
    train_IoU = last_model['train_IoU']

    accuracy_validation = last_model['val_accuracy']
    loss_validation = last_model['val_loss']
    val_IoU = last_model['val_IoU']  
    
    f1 = plt.figure(1)
    plt.plot(accuracy_training, label = "Training Accuracy")
    plt.plot(accuracy_validation, label = "Validation Accuracy")
    plt.title("Accuracy Over Epochs for Transfer Segmentation Model ")
    plt.grid()
    plt.legend()
    f2 = plt.figure(2)
    plt.plot(loss_training, label = "Training Loss")
    plt.plot(loss_validation, label = "Validation Loss")
    plt.title("Loss Over Epochs for Segmentation Model Created From Scratch")
    plt.grid()
    plt.legend()
    f3 = plt.figure(3)
    plt.plot(train_IoU, label = "Training mean IoU")
    plt.plot(val_IoU, label = "Validation mean IoU")
    plt.title("IoU Over Epochs for Segmentation Model Created From Scratch")
    plt.grid()
    plt.legend()
    plt.show()


def load_and_display_cm(path=".\\CM\\REALcm_Test_Segmentation.npy", class_names=None, normalize=True, cmap='Blues'):
    cm = np.load(path).T  # <-- transpose to swap x and y axes
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)  # normalize along new "true" axis

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix For Segmentation Model From Scratch")
    plt.colorbar()

    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    else:
        plt.xticks(np.arange(cm.shape[1]))
        plt.yticks(np.arange(cm.shape[0]))

    plt.xlabel('True label')      # flipped position
    plt.ylabel('Predicted label') # flipped position
    plt.tight_layout()
    plt.show()


def plot_VQA_metrics():
    last_model = torch.load('.\\Models\\last_VQA_model.pth', weights_only=False)
    accuracy_training = last_model['train_acc']
    loss_training = last_model['train_loss']
    accuracy_validation = last_model['val_accuracy']
    loss_validation = last_model['val_loss']
    f1 = plt.figure(1)
    plt.plot(accuracy_training, label = "Training Accuracy")
    plt.plot(accuracy_validation, label = "Validation Accuracy")
    plt.title("Accuracy Over Epochs for VQA Model")
    plt.grid()
    plt.legend()
    f2 = plt.figure(2)
    plt.plot(loss_training, label = "Training Loss")
    plt.plot(loss_validation, label = "Validation Loss")
    plt.title("Loss Over Epochs for VQA Model")
    plt.grid()
    plt.legend()
    plt.show()


### function to get the predicted result from the models given a sample
def classify_img(img):
    best_c_net = torch.load(".\\Models\\best_resnet_transfer_classification_model.pth", map_location='cpu', weights_only=False)
    net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    net.fc = nn.Sequential(
        nn.Linear(net.fc.in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, 2)
    )
    net.load_state_dict(best_c_net['model_state_dict'])
    net.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]) 
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = net(img_tensor)
    pred = torch.argmax(out, dim=1).cpu().numpy()[0]
    return pred
    
def color_segmented_map(img):
    best_s_net = torch.load(".\\Models\\2TPSP_WDice_best_segmentation_model.pth", map_location='cpu', weights_only=False)
    s_net = Resnet_Transfer_SegNet()
    s_net.load_state_dict(best_s_net['model_state_dict'])
    s_net.eval()

    transform = transforms.Compose([
        transforms.Resize((360, 360)),
        transforms.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ]) 
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = s_net(img_tensor)
    color_img = img_to_color(out)
    return color_img    

def img_to_color(img):
    pred = torch.argmax(img, dim=1).cpu().numpy()[0]
    #st.write("Unique predicted classes:", np.unique(pred))
    h, w = pred.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_to_color.items():
        mask = pred == class_id
        color_img[mask] = np.array(color, dtype=np.uint8)

    return color_img

def convert_to_grayscale(path, type):
    all_images = []
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        img_gs = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)

        for color, class_id in color_to_class.items():
            mask = np.all(img == color, axis=-1)
            img_gs[mask] = class_id
        all_images.append(img_gs)
    all_images = np.array(all_images)
    np.save(f".\\Dataset\\Segmentation_Mask_GreyScale\\{type}_grayscaleMasks.npy", all_images)
    print(f"finished {type}")
    if type == "train":
        unique, counts = np.unique(all_images, return_counts=True)
        class_pixel_counts = dict(zip(unique, counts))
        print("Pixel count per class:")
        for class_id, count in sorted(class_pixel_counts.items()):
            print(f"Class {class_id}: {count} pixels")

###
# for VQA #
###

def query_VQA(img, file_name):
    all_answers = []
    with open(".\\Dataset\\VQA_Dataset\\test_annotations.json", 'r') as f:
            qa_test_pairs = json.load(f)
    questions = [item["Question"] for item in qa_test_pairs if item["Image_ID"].lower()== file_name.lower()]
    all_gt = [item["Ground_Truth"] for item in qa_test_pairs if item["Image_ID"].lower()== file_name.lower()]
    print(questions)
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])
    best_vqa_net = torch.load(".\\Models\\best_vqa_model.pth", map_location='cpu', weights_only=False)
    net = VQA_Net()
    net.load_state_dict(best_vqa_net['model_state_dict'])
    img_tensor = transform(img).unsqueeze(0)
    with open(".\\Dataset\\VQA_Dataset\\word_to_token.json", 'r') as f:
        tokenizer = json.load(f)
    with open(".\\Dataset\\VQA_Dataset\\class_to_label.json", 'r') as f:
        answer_dict = json.load(f)
    net.eval()
    for question in questions:
        question = question.lower().strip()
        question_words = question.translate(str.maketrans('', '', string.punctuation)).split()
        q_tokens = [tokenizer.get(word, tokenizer["unk"]) for word in question_words]
        if len(q_tokens) < 20:
            q_tokens += [0] * (20 - len(q_tokens)) 
        else:
            q_tokens = q_tokens[:20]
        q_tokens = torch.tensor(q_tokens, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            out = net(img_tensor, q_tokens)
        pred = torch.argmax(out)
        label_to_class = {v: k for k, v in answer_dict.items()}
        answer = label_to_class[pred.item()]
        all_answers.append(answer)
    return questions, all_answers, all_gt


### util function used to fix the directory of images in VQA dataset
def fix_VQA_dataset(json_path, type):
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    new_base_path = f".\\Dataset\\FloodNet-Supervised_v1.0\\{type}\\{type}-org-img"
    for item in json_file:
        file_name = os.path.basename(item['Image_dir'])
        item['Image_dir'] = os.path.join(new_base_path, file_name)
    output_path =  os.path.join(os.path.dirname(json_path), "updated_" + os.path.basename(json_path))
    with open(output_path, "w") as f:
        json.dump(json_file, f, indent=4)



    


if __name__ == '__main__':
    plot_VQA_metrics()
