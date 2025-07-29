import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
#local imports
from datasets.classification_dataset import ClassificationDataset
from networks.classification_network import ClassificationNetwork
from utils import compute_positional_weight, compute_label_weight

###
#   This file contains the training pipeline for classification models
#   formatted as callable functions in main
###
def classification_pipeline():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    batch_size = 16

    train_ds = ClassificationDataset("./Dataset/FloodNet-Supervised_v1.0/train/train-org-img", "./Dataset/y_classification/y_train.npy", transform)
    val_ds = ClassificationDataset("./Dataset/FloodNet-Supervised_v1.0/val/val-org-img", "./Dataset/y_classification/y_val.npy", transform)
    test_ds = ClassificationDataset("./Dataset/FloodNet-Supervised_v1.0/test/test-org-img", "./Dataset/y_classification/y_test.npy", transform)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=5
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=5
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=5
    )
    
    train_pos_weight = compute_positional_weight(train_ds.y_labels)
    print("Positional weight:", train_pos_weight.item())
    val_pos_weight = compute_positional_weight(val_ds.y_labels)
    test_pos_weight = compute_positional_weight(test_ds.y_labels)
    
    lr = 0.001
    patience = 15
    delta = 0.005
    l_fun = nn.BCEWithLogitsLoss(pos_weight = train_pos_weight)
    device = 'cuda'
    epoch_train_losses = []
    epoch_train_accuracy = []
    epoch_val_losses = []
    epoch_val_accuracy = []
    best_loss = None
    net = ClassificationNetwork()
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr, weight_decay= 0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, threshold=delta
        )
    #training loop
    print("Training Started!")
    for cur_epoch in range(50):
        
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_correct = 0
        for inp, gt in tqdm(train_dl, desc=f"Epoch {cur_epoch}"):
            inp = inp.to(device)
            gt = gt.to(device)

            out = net(inp)
            loss = l_fun(out, gt.float().unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inp.size(0)
            prob = torch.sigmoid(out)
            pred = (prob > 0.5).long()
            correct = (pred.squeeze() == gt).sum().item()
            epoch_correct += correct
            epoch_samples += inp.size(0)
        epoch_loss = epoch_loss/epoch_samples
        epoch_accuracy = epoch_correct/epoch_samples
        epoch_train_losses.append(epoch_loss)
        epoch_train_accuracy.append(epoch_accuracy)

        print(f"Epoch {cur_epoch}")
        print(f" Training - loss: {epoch_loss}, accuracy: {epoch_accuracy}")
        val_loss, val_acc = validation(val_dl, net, device, val_pos_weight) 
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
            }, "./Models/best_classification_model.pth")
        else:
            patience -=1
            print(f"Patience down to {patience}")
            if patience <= 0: 
                print("Early stopping triggered!")
                break
        scheduler.step(val_loss)
    print("Training Stopped, next metrics will be on the test set")
    test_loss, test_acc = validation(test_dl, net, device, test_pos_weight)
    torch.save({
            'epoch': cur_epoch,
            'model_state_dict' : net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            'accuracy': val_acc,
            'train_loss':np.array(epoch_train_losses),
            'train_acc': np.array(epoch_train_accuracy),
            'val_loss': np.array(epoch_val_losses),
            'val_accuracy': np.array(epoch_val_accuracy),
            'test_acc' : np.array(test_acc),
            'test_loss': np.array(test_loss)
        }, "./Models/last_classification_model.pth")

def resnet_transfer_pipeline():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    batch_size = 16

    train_ds = ClassificationDataset("./Dataset/FloodNet-Supervised_v1.0/train/train-org-img", "./Dataset/y_classification/y_train.npy", transform)
    val_ds = ClassificationDataset("./Dataset/FloodNet-Supervised_v1.0/val/val-org-img", "./Dataset/y_classification/y_val.npy", transform)
    test_ds = ClassificationDataset("./Dataset/FloodNet-Supervised_v1.0/test/test-org-img", "./Dataset/y_classification/y_test.npy", transform)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=5
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=5
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=5
    )
    
    train_weight = compute_label_weight(train_ds.y_labels)
    val_weight = compute_label_weight(val_ds.y_labels)
    test_weight = compute_label_weight(test_ds.y_labels)
    
    lr = 0.001
    patience = 10
    delta = 0.005
    l_fun = nn.CrossEntropyLoss(weight=train_weight)
    device = 'cuda'
    epoch_train_losses = []
    epoch_train_accuracy = []
    epoch_val_losses = []
    epoch_val_accuracy = []
    best_loss = None
    net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    #No softmax as it is handled by CrossEntropyLoss
    net.fc = nn.Sequential(
        nn.Linear(net.fc.in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, 2)
    )
    for name, param in net.named_parameters():
        if not name.startswith('fc'):
            param.requires_grad = False
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    
    #training loop
    print("Training Started!")
    for cur_epoch in range(40):
        
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_correct = 0
        for inp, gt in tqdm(train_dl, desc=f"Epoch {cur_epoch}"):
            inp = inp.to(device)
            gt = gt.to(device)

            out = net(inp)
            loss = l_fun(out, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inp.size(0)
            pred = torch.argmax(out, dim = 1)
            correct = (pred.squeeze() == gt).sum().item()
            epoch_correct += correct
            epoch_samples += inp.size(0)
        epoch_loss = epoch_loss/epoch_samples
        epoch_accuracy = epoch_correct/epoch_samples
        epoch_train_losses.append(epoch_loss)
        epoch_train_accuracy.append(epoch_accuracy)

        print(f"Epoch {cur_epoch}")
        print(f" Training - loss: {epoch_loss}, accuracy: {epoch_accuracy}")
        val_loss, val_acc = transfer_validation(val_dl, net, device, val_weight) 
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
                'accuracy': val_acc
            }, "./Models/best_resnet_transfer_classification_model.pth")
        else:
            patience -=1
            print(f"Patience down to {patience}")
            if patience <= 0: 
                print("Early stopping triggered!")
                break
    print("Training Stopped, saving last model...")
    #Test
    print("Next metrics will be on the test set")
    test_loss, test_acc = transfer_validation(test_dl, net, device, test_weight)
    torch.save({
            'epoch': cur_epoch,
            'model_state_dict' : net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            'accuracy': val_acc,
            'train_loss':np.array(epoch_train_losses),
            'train_acc': np.array(epoch_train_accuracy),
            'val_loss': np.array(epoch_val_losses),
            'val_accuracy': np.array(epoch_val_accuracy),
            'test_acc' : np.array(test_acc),
            'test_loss': np.array(test_loss)
        }, "./Models/last_resnet_transfer_classification_model.pth")



def validation(dl, net, device, pos_weight):
    loss_fun = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
    net.eval()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for inp, gt in dl:
        inp = inp.to(device)
        gt = gt.to(device)
        with torch.no_grad():
            out = net(inp)
        #BCE loss expects float and [batch_size, 1] dims
        loss = loss_fun(out, gt.float().unsqueeze(1))
        #getting the total loss for the batch, used to get the total average after
        total_loss += loss.item() * inp.size(0)
        prob = torch.sigmoid(out)
        pred = (prob > 0.5).long()
        correct = (pred.squeeze() == gt).sum().item()
        total_correct += correct
        total_samples += inp.size(0) 
    
    avg_loss = total_loss/total_samples
    accuracy = total_correct/total_samples 
    print(f" Validation set - loss: {avg_loss}, accuracy: {accuracy}")
    net.train()
    return avg_loss, accuracy


def transfer_validation(dl, net, device, weight):
    loss_fun = nn.CrossEntropyLoss(weight = weight)
    net.eval()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for inp, gt in dl:
        inp = inp.to(device)
        gt = gt.to(device)
        with torch.no_grad():
            out = net(inp)
        #CrossEntropyloss expects float and [batch_size] dims
        loss = loss_fun(out, gt)
        #getting the total loss for the batch, used to get the total average after
        total_loss += loss.item() * inp.size(0)
        pred = torch.argmax(out, dim = 1)
        correct = (pred.squeeze() == gt).sum().item()
        total_correct += correct
        total_samples += inp.size(0) 
    
    avg_loss = total_loss/total_samples
    accuracy = total_correct/total_samples 
    print(f" Validation set - loss: {avg_loss}, accuracy: {accuracy}")
    net.train()
    return avg_loss, accuracy


def network_test():
    device = 'cuda'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    batch_size = 16
    test_ds = ClassificationDataset("./Dataset/FloodNet-Supervised_v1.0/test/test-org-img", "./Dataset/y_classification/y_test.npy", transform)
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=5
    )
    test_weight = compute_label_weight(test_ds.y_labels)
    transfer_model = torch.load('.\\Models\\best_resnet_transfer_classification_model.pth')
    t_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    #No softmax as it is handled by CrossEntropyLoss
    t_net.fc = nn.Sequential(
        nn.Linear(t_net.fc.in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, 2)
    )
    t_net.load_state_dict(transfer_model['model_state_dict']) 
    t_net = t_net.to(device)
    print("Transfer")
    transfer_validation(test_dl, t_net, device, test_weight)

if __name__ == "__main__":
    network_test()