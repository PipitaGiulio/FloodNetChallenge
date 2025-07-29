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
from datasets.segmentation_dataset import SegmentationDataset
from networks.segmentation_network import SegmentationNet, PSP_SegmentationNet, Resnet_Transfer_SegNet 
from utils import compute_segmentation_weights

import time

###
#   This file contains the training pipeline for segmentation models
#   formatted as callable functions in main
###
def transfer_segmentation_pipeline():
    #resnet101
    epochs = 80
    batch_size = 2
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(713, 713), scale = (0.5, 1.0), interpolation=cv2.INTER_NEAREST),
        A.Rotate(limit=45, p=0.5),
        A.HorizontalFlip(p = 0.5),
        A.VerticalFlip(p = 0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.GaussianBlur(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
        A.ToTensorV2()
    ])
    val_test_transform = A.Compose([
        A.Resize(713, 713),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
        A.ToTensorV2()
    ])

    train_ds = SegmentationDataset(".\\Dataset\\FloodNet-Supervised_v1.0\\train\\train-org-img", ".\\Dataset\\FloodNet-Supervised_v1.0\\train\\train-label-img", train_transform)
    val_ds = SegmentationDataset(".\\Dataset\\FloodNet-Supervised_v1.0\\val\\val-org-img", ".\\Dataset\\FloodNet-Supervised_v1.0\\val\\val-label-img", val_test_transform)
    test_ds = SegmentationDataset(".\\Dataset\\FloodNet-Supervised_v1.0\\test\\test-org-img", ".\\Dataset\\FloodNet-Supervised_v1.0\\test\\test-label-img", val_test_transform)




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
    net = Resnet_Transfer_SegNet()
    net.to(device)
    optimizer = SGD(
        net.parameters(),
        lr = lr,
        momentum = 0.9,
        weight_decay=0.00001
    )
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer=optimizer,
        total_iters=epochs,
        power=0.9,
        last_epoch=-1
    )

    

    train_weight = torch.load('.\\segmentation_weights\\train_weights.pt').to(device)
    val_weight = torch.load('.\\segmentation_weights\\val_weights.pt').to(device)
    test_weight = torch.load('.\\segmentation_weights\\test_weights.pt').to(device)
    
    loss_fun = nn.CrossEntropyLoss(weight=train_weight)
    dice_fun  = DiceScore(num_classes=10, average="macro")
    best_loss = None
    patience = 30
    delta = 0.005
    train_IoUs = []
    val_IoUs = []

    all_train_losses = []
    all_train_flood_accuracy = []
    all_train_accuracy = []
    all_train_IoU = []
    all_val_losses = []
    all_val_flood_accuracy = []
    all_val_accuracy = []
    all_val_IoU = []
    print("Training Started!")
    t0 = time.time()
    alpha = 0.3
    beta = 0.7

    for cur_epoch in range(epochs):
        if cur_epoch > 40:
            delta = 0.001
        elif cur_epoch > 20:
            delta = 0.002
        all_preds = []
        all_targets = []
        epoch_loss = 0
        epoch_samples = 0
        for inp, gt in tqdm(train_dl, desc=f"Epoch {cur_epoch}"):
            
            inp = inp.to(device)
            gt = gt.to(device)

            out = net(inp)

            preds = torch.argmax(out, dim = 1)
            preds = preds.view(-1).cpu().numpy() 
            masks = gt.view(-1).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(masks)

            
            loss_CE = loss_fun(out, gt)
            gt_one_hot = one_hot(gt, num_classes=10).permute(0, 3, 1, 2).float()
            loss_dice = 1 - dice_fun(torch.softmax(out, dim=1), gt_one_hot)
            loss = alpha * loss_CE + beta * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inp.size(0)
            epoch_samples += inp.size(0)

        scheduler.step()
        epoch_loss = epoch_loss/epoch_samples
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        _, IoU, epoch_mean_IoU, epoch_mean_flood_accuracy, epoch_mean_accuracy = compute_mIoU_accuracy(all_preds, all_targets)
        train_IoUs.append(IoU)
        all_train_losses.append(epoch_loss)
        all_train_IoU.append(epoch_mean_IoU)
        all_train_flood_accuracy.append(epoch_mean_flood_accuracy)
        all_train_accuracy.append(epoch_mean_accuracy)
        print(f"Epoch {cur_epoch}, loss: {epoch_loss:.3f}, m_IoU: {epoch_mean_IoU:.3f}, m_flood_accuracy: {epoch_mean_flood_accuracy:.3f}, m_accuracy: {epoch_mean_accuracy:.3f}")

        val_Iou, val_mean_Iou, val_mean_flood_acc, val_mean_accuracy, val_loss = validation(val_dl, net, device, train_weight, "Validation")
        all_val_losses.append(val_loss)
        val_IoUs.append(val_Iou)
        all_val_IoU.append(val_mean_Iou)
        all_val_flood_accuracy.append(val_mean_flood_acc)
        all_val_accuracy.append(val_mean_accuracy)
        if best_loss is None or val_mean_Iou > best_loss + delta:
            best_loss = val_mean_Iou
            if patience < 5 and cur_epoch > 30:
                patience +=1
                print(f"Raised Patience to {patience}") 
            torch.save({
                'epoch': cur_epoch,
                'model_state_dict' : net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'flood_accuracy': val_mean_flood_acc,
                'mean_accuracy': val_mean_accuracy,
                'IoU': val_mean_Iou,
                'train_loss':np.array(all_train_losses),
                'train_flood_acc': np.array(all_train_flood_accuracy),
                'train_accuracy': np.array(all_train_accuracy),
                'train_IoU': np.array(all_train_IoU),
                'val_loss': np.array(all_val_losses),
                'val_flood_accuracy': np.array(all_val_flood_accuracy),
                'val_accuracy' : np.array(all_val_accuracy),
                'val_IoU': np.array(all_val_IoU)
            }, "./Models/2T-PSP_WDice_best_segmentation_model.pth")
        else:
            patience -= 1
            print(f"Patience down to {patience}")
            if patience <= 0:
                print("Patience reached 0, stopping due to callback")
                break
    print(f"Training Stopped after {((time.time() - t0)/60):2f}, saving last model...")
    torch.save({
            'epoch': cur_epoch,
            'model_state_dict' : net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            'accuracy': val_mean_flood_acc,
            'IoU': val_mean_Iou,
            'train_loss':np.array(all_train_losses),
            'train_flood_acc': np.array(all_train_flood_accuracy),
            'train_accuracy': np.array(all_train_accuracy),
            'train_IoU': np.array(all_train_IoU),
            'val_loss': np.array(all_val_losses),
            'val_flood_accuracy': np.array(all_val_flood_accuracy),
            'val_accuracy' : np.array(all_val_accuracy),
            'val_IoU': np.array(all_val_IoU),
            'class_val_Iou': np.stack(val_IoUs),
            'class_train_Iou': np.stack(train_IoUs)
        }, "./Models/2TPSP_WDice_last_segmentation_model.pth")
    
    print("Start Test")
    validation(test_dl, net, device, train_weight, "Test")


def PSP_like_segmentation_pipeline():
    epochs = 120
    batch_size = 4
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(360, 360), scale = (0.5, 1.0), interpolation=cv2.INTER_NEAREST),
        A.Rotate(limit=45, p=0.5),
        A.HorizontalFlip(p = 0.5),
        A.VerticalFlip(p = 0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.GaussianBlur(p=0.2),
        A.CoarseDropout(
            num_holes_range=(4, 12),
            hole_height_range=(0.05, 0.1),
            hole_width_range=(0.05, 0.1),
            fill=0,           
            fill_mask=0,      
            p=0.5
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
        A.ToTensorV2()
    ])
    val_test_transform = A.Compose([
        A.Resize(360, 360),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
        A.ToTensorV2()
    ])

    train_ds = SegmentationDataset(".\\Dataset\\FloodNet-Supervised_v1.0\\train\\train-org-img", ".\\Dataset\\FloodNet-Supervised_v1.0\\train\\train-label-img", train_transform)
    val_ds = SegmentationDataset(".\\Dataset\\FloodNet-Supervised_v1.0\\val\\val-org-img", ".\\Dataset\\FloodNet-Supervised_v1.0\\val\\val-label-img", val_test_transform)
    test_ds = SegmentationDataset(".\\Dataset\\FloodNet-Supervised_v1.0\\test\\test-org-img", ".\\Dataset\\FloodNet-Supervised_v1.0\\test\\test-label-img", val_test_transform)
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
        num_workers=8
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8
    )
    
    lr = 0.01
    device = 'cuda'
    net = PSP_SegmentationNet()
    net.to(device)
    optimizer = SGD(
        net.parameters(),
        lr = lr,
        momentum = 0.9,
        weight_decay=0.00001
    )
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer=optimizer,
        total_iters=epochs,
        power=0.9,
        last_epoch=-1
    )
    train_weight = torch.load('.\\segmentation_weights\\train_weights.pt').to(device)
    val_weight = torch.load('.\\segmentation_weights\\val_weights.pt').to(device)
    test_weight = torch.load('.\\segmentation_weights\\test_weights.pt').to(device)
    
    loss_fun = nn.CrossEntropyLoss(weight=train_weight)
    dice_fun  = DiceScore(num_classes=10, average="macro")
    best_loss = None
    patience = 35
    delta = 0.005
    alpha = 0.3
    beta = 0.7
    train_IoUs = []
    val_IoUs = []

    all_train_losses = []
    all_train_flood_accuracy = []
    all_train_accuracy = []
    all_train_IoU = []
    all_val_losses = []
    all_val_flood_accuracy = []
    all_val_accuracy = []
    all_val_IoU = []
    print("Training Started!")
    t0 = time.time()
    

    for cur_epoch in range(epochs):
        if cur_epoch > 60:
            delta = 0.001
        elif cur_epoch > 30:
            delta = 0.002
        all_preds = []
        all_targets = []
        epoch_loss = 0
        epoch_samples = 0
        for inp, gt in tqdm(train_dl, desc=f"Epoch {cur_epoch}"):
            
            inp = inp.to(device)
            gt = gt.to(device)

            out = net(inp)

            preds = torch.argmax(out, dim = 1)
            preds = preds.view(-1).cpu().numpy() 
            masks = gt.view(-1).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(masks)

            loss_CE = loss_fun(out, gt)
            gt_one_hot = one_hot(gt, num_classes=10).permute(0, 3, 1, 2).float()
            loss_dice = 1 - dice_fun(torch.softmax(out, dim=1), gt_one_hot)
            loss = alpha * loss_CE + beta * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inp.size(0)
            epoch_samples += inp.size(0)

        scheduler.step()
        epoch_loss = epoch_loss/epoch_samples
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        _, IoU, epoch_mean_IoU, epoch_mean_flood_accuracy, epoch_mean_accuracy = compute_mIoU_accuracy(all_preds, all_targets)
        train_IoUs.append(IoU)
        all_train_losses.append(epoch_loss)
        all_train_IoU.append(epoch_mean_IoU)
        all_train_flood_accuracy.append(epoch_mean_flood_accuracy)
        all_train_accuracy.append(epoch_mean_accuracy)
        print(f"Epoch {cur_epoch}, loss: {epoch_loss:.3f}, m_IoU: {epoch_mean_IoU:.3f}, m_flood_accuracy: {epoch_mean_flood_accuracy:.3f}, m_accuracy: {epoch_mean_accuracy:.3f}")

        val_Iou, val_mean_Iou, val_mean_flood_acc, val_mean_accuracy, val_loss = validation(val_dl, net, device, train_weight, "Validation")
        all_val_losses.append(val_loss)
        val_IoUs.append(val_Iou)
        all_val_IoU.append(val_mean_Iou)
        all_val_flood_accuracy.append(val_mean_flood_acc)
        all_val_accuracy.append(val_mean_accuracy)
        if best_loss is None or val_mean_Iou > best_loss + delta:
            best_loss = val_mean_Iou
            if patience < 5 and cur_epoch > 30:
                patience +=1
                print(f"Raised Patience to {patience}") 
            torch.save({
                'epoch': cur_epoch,
                'model_state_dict' : net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'flood_accuracy': val_mean_flood_acc,
                'mean_accuracy': val_mean_accuracy,
                'IoU': val_mean_Iou,
                'train_loss':np.array(all_train_losses),
                'train_flood_acc': np.array(all_train_flood_accuracy),
                'train_accuracy': np.array(all_train_accuracy),
                'train_IoU': np.array(all_train_IoU),
                'val_loss': np.array(all_val_losses),
                'val_flood_accuracy': np.array(all_val_flood_accuracy),
                'val_accuracy' : np.array(all_val_accuracy),
                'val_IoU': np.array(all_val_IoU)
            }, "./Models/PSP2_like_best_segmentation_model.pth")
        else:
            patience -= 1
            print(f"Patience down to {patience}")
            if patience <= 0:
                print("Patience reached 0, stopping due to callback")
                break
    print(f"Training Stopped, Start Test")
    test_Iou, test_mean_IoU, _, test_accuracy, test_loss = validation(test_dl, net, device, train_weight, "Test")
    print("Saving last model")
    torch.save({
            'epoch': cur_epoch,
            'model_state_dict' : net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            'accuracy': val_mean_flood_acc,
            'IoU': val_mean_Iou,
            'train_loss':np.array(all_train_losses),
            'train_flood_acc': np.array(all_train_flood_accuracy),
            'train_accuracy': np.array(all_train_accuracy),
            'train_IoU': np.array(all_train_IoU),
            'val_loss': np.array(all_val_losses),
            'val_flood_accuracy': np.array(all_val_flood_accuracy),
            'val_accuracy' : np.array(all_val_accuracy),
            'val_IoU': np.array(all_val_IoU),
            'class_val_Iou': np.stack(val_IoUs),
            'class_train_Iou': np.stack(train_IoUs),
            'test_loss': np.array(test_loss),
            'test_accuracy': np.array(test_accuracy),
            'test_IoU': np.array(test_Iou),
            'test_mean_Iou': np.array(test_mean_IoU)
        }, "./Models/PSP2_like_last_segmentation_model.pth")
    

def compute_mIoU_accuracy(preds, gt):
    #using the confusion matrix to 
    cm = confusion_matrix(gt, preds, labels=range(10))
    TP = np.diag(cm)
    FP = cm.sum(axis = 0) - TP
    FN = cm.sum(axis = 1) - TP
    TN = cm.sum() - (TP + FP + FN)

    denom = TP + FP + FN
    valid = denom > 0
    IoU = np.zeros_like(TP, dtype=np.float64)
    IoU[valid] = TP[valid] / denom[valid]
    mean_IoU = IoU[valid].mean() if np.any(valid) else 0.0

    flood_accuracy = (TP + TN) / (TP + TN + FP + FN)
    mean_flood_accuracy = np.mean(flood_accuracy)
    accuracy = TP / (TP + FN)
    mean_accuracy = np.mean(accuracy)
    return cm, IoU, mean_IoU, mean_flood_accuracy, mean_accuracy


def validation(dl, net, device, weights, type):
    l_fun = nn.CrossEntropyLoss(weight=weights)
    d_fun = DiceScore(num_classes=10, average='macro')
    net.eval()
    all_preds = []
    all_targets = []
    avg_loss = 0
    samples = 0
    for inp, gt in dl:
        inp = inp.to(device)
        gt = gt.to(device)

        with torch.no_grad():
            out = net(inp)
        preds = torch.argmax(out, dim = 1)
        preds = preds.view(-1).cpu().numpy() 
        masks = gt.view(-1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(masks)

        l_CE = l_fun(out, gt)
        gt_one_hot = one_hot(gt, num_classes=10).permute(0, 3, 1, 2).float()
        l_dice = d_fun(torch.softmax(out, dim=1), gt_one_hot)
        loss = 0.3 * l_CE + 0.7 * l_dice
        avg_loss += loss.item() * inp.size(0)
        samples += inp.size(0)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    avg_loss = avg_loss/samples
    cm ,val_Iou, val_mean_IoU, val_mean_flood_accuracy, val_accuracy = compute_mIoU_accuracy(all_preds, all_targets)
    if type == 'Test':
        np.save(".\\CM\\REALcm_Test_Segmentation.npy", cm)
    print(f" {type}, loss: {avg_loss:.3f}, m_IoU: {val_mean_IoU:.3f}, m_flood_accuracy: {val_mean_flood_accuracy:.3f}, m_accuracy: {val_accuracy:.3f}")
    net.train()
    return val_Iou, val_mean_IoU, val_mean_flood_accuracy, val_accuracy, avg_loss


