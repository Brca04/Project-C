#!/usr/bin/env python
# coding: utf-8

import torch
import multiprocessing
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# --- 1. SETUP & DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10 
# Imagenette has 10 classes

train_losses = []
val_losses = []
train_accs = []
val_accs = []

# --- 2. MODEL: EFFICIENTNET-B1 ---
model = models.efficientnet_b1(weights=None)

# Replace classifier (EfficientNet style)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

model = model.to(device)

for p in model.parameters():
    p.requires_grad = True

optimiser = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

criterion = nn.CrossEntropyLoss()

# --- 3. DATASET: IMAGENETTE (LOCAL) ---
# EfficientNet-B1 requires 240x240 input size

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(240),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406), 
        std=(0.229, 0.224, 0.225),
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(240),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406), 
        std=(0.229, 0.224, 0.225),
    ),
])

# Your local path
data_dir = r"C:\Users\Bruno\Desktop\FER\PROJEKT R\CNN\data\imagenette2\imagenette2"
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(root=val_dir, transform=val_transform)

# EfficientNet is lighter than VGG → 64 is OK, reduce if OOM
batch_size = 96 

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# --- 4. EVALUATION FUNCTION ---
def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            prediction = outputs.argmax(dim=1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

    val_loss = val_loss / total
    val_acc = correct / total
    return val_loss, val_acc

# --- 5. TRAINING LOOP ---
if __name__ == "__main__":
    multiprocessing.freeze_support()

    epohs = 25

    for i in range(epohs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * images.size(0)
            prediction = outputs.argmax(dim=1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Epoch {i+1}/{epohs} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f}")

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {i+1}/{epohs} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    # Final Test (using val set)
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            prediction = outputs.argmax(dim=1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

    test_loss = test_loss / total
    test_acc = correct / total
    print(f"Final Test Result (on Val Set): test_loss={test_loss:.4f} | test_acc={test_acc:.4f}")

    epochs_range = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train loss")
    plt.plot(epochs_range, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss (EfficientNet-B1)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(epochs_range, train_accs, label="Train accuracy")
    plt.plot(epochs_range, val_accs, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy (EfficientNet-B1)")
    plt.legend()
    plt.grid(True)
    plt.show()
