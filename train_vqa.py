import pandas as pd
import numpy as np
import torch
from torch.optim import SGD
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import os
from torchvision import transforms
import albumentations as A
from sklearn.metrics import confusion_matrix 
import cv2
from torch.nn.functional import one_hot
from torchmetrics.segmentation import DiceScore
#local imports
from networks.vqa_network import VQA_Net, transfer_VQA
from datasets.VQA_dataset import VQADataset
###
#   This file contains the training pipeline for VQA model
#   formatted as callable function in main
###
def vqa_pipeline():
    batch_size = 8
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    train_ds = VQADataset(".\\Dataset\\VQA_Dataset\\updated_train_annotations.json", transform)
    test_ds = VQADataset(".\\Dataset\\VQA_Dataset\\updated_test_annotations.json", transform)
    val_ds = VQADataset(".\\Dataset\\VQA_Dataset\\updated_valid_annotations.json", transform)
    
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    lr = 0.01
    device = 'cuda'
    #net = VQA_Net()
    net = transfer_VQA()
    net.to(device)
    optimizer = SGD(
        net.parameters(),
        lr = lr,
        momentum = 0.9,
        weight_decay=0.00001
    )
    loss_fun = nn.CrossEntropyLoss()
    epoch_train_losses = []
    epoch_train_accuracy = []
    epoch_val_losses = []
    epoch_val_accuracy = []
    best_loss = None
    delta = 0.005
    patience = 30
    print("Training Started!")
    for cur_epoch in range(45):
        
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_correct = 0
        for img_inp, q_inp, gt in tqdm(train_dl, desc=f"Epoch {cur_epoch}"):
            img_inp = img_inp.to(device)
            q_inp = q_inp.to(device)
            gt = gt.to(device)

            out = net(img_inp, q_inp)
            loss = loss_fun(out, gt.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * q_inp.size(0)
            pred = torch.argmax(out, dim=1)
            correct = (pred == gt).sum().item()
            epoch_correct += correct
            epoch_samples += q_inp.size(0)
        epoch_loss = epoch_loss/epoch_samples
        epoch_accuracy = epoch_correct/epoch_samples
        epoch_train_losses.append(epoch_loss)
        epoch_train_accuracy.append(epoch_accuracy)

        print(f"Epoch {cur_epoch}")
        print(f" Training - loss: {epoch_loss}, accuracy: {epoch_accuracy}")
        val_loss, val_acc = validation(val_dl, net, device) 
        epoch_val_losses.append(val_loss)
        epoch_val_accuracy.append(val_acc)
        if best_loss == None or val_loss < best_loss - delta:
            best_loss = val_loss
            if patience < 2 and cur_epoch > 5:
                patience +=1
                print(f"Raised Patience to {patience}") 
            torch.save({
                'epoch': cur_epoch,
                'model_state_dict' : net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'accuracy': val_acc,
                'train_loss':np.array(epoch_train_losses),
                'train_acc': np.array(epoch_train_accuracy),
                'val_loss': np.array(epoch_val_losses),
                'val_accuracy': np.array(epoch_val_accuracy)
            }, "./Models/best_Transfer_VQA_model.pth")
        else:
            patience -=1
            print(f"Patience down to {patience}")
            if patience <= 0: 
                print("Early stopping triggered!")
                break
    print("Training Stopped, saving last model...")
    torch.save({
            'epoch': cur_epoch,
            'model_state_dict' : net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'accuracy': val_acc,
            'train_loss':np.array(epoch_train_losses),
            'train_acc': np.array(epoch_train_accuracy),
            'val_loss': np.array(epoch_val_losses),
            'val_accuracy': np.array(epoch_val_accuracy)
        }, "./Models/last_Transfer_VQA_model.pth")
    #Test
    print("Next metrics will be on the test set")
    validation(test_dl, net, device)
    
def validation(dl, net, device):
    l_fun = nn.CrossEntropyLoss()
    net.eval()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    for img_inp, q_inp, gt in tqdm(dl, desc=f"Validation"):
            img_inp = img_inp.to(device)
            q_inp = q_inp.to(device)
            gt = gt.to(device)
            with torch.no_grad():
                out = net(img_inp, q_inp)
            loss = l_fun(out, gt.long())
            total_loss += loss.item() * q_inp.size(0)
            pred = torch.argmax(out, dim=1)
            correct = (pred == gt).sum().item()
            total_correct += correct
            total_samples += q_inp.size(0)
    avg_loss = total_loss/total_samples
    accuracy = total_correct/total_samples 
    print(f" Validation set - loss: {avg_loss}, accuracy: {accuracy}")
    net.train()
    return avg_loss, accuracy


###utils function to test the model
def test_VQA():
    last_model = torch.load('.\\Models\\best_VQA_model.pth', weights_only=False)
    batch_size = 8
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    test_ds = VQADataset(".\\Dataset\\VQA_Dataset\\updated_test_annotations.json", transform)
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    device = 'cuda'
    #net = VQA_Net()
    net = transfer_VQA()
    net.load_state_dict(last_model['model_state_dict'])
    net.to(device)
    validation(test_dl, net, device)

if __name__ == '__main__':
    test_VQA()

