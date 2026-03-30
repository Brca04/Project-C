#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import multiprocessing
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Subset
import matplotlib.pyplot as plt


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 100

train_losses = []
val_losses = []
train_accs = []
val_accs = []


# In[ ]:


model = models.resnet18(weights=None) #treniranje iz nule

model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity() # mijenjamo conv1 jer kako bi se prilagodili CIFAR velicini slika
model.fc = nn.Linear(model.fc.in_features, num_classes)


model = model.to(device)

for p in model.parameters():
    p.requires_grad = True

optimiser = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

# scheduler = torch.optim.lr_scheduler.StepLR(
#     optimiser, step_size=30, gamma=0.1
# )

criterion = nn.CrossEntropyLoss()


# In[ ]:


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
])

base_dataset = datasets.CIFAR100(
    root="./data",
    train=True,
    download=True,
    transform=None
)

N = len(base_dataset)
perm_img = torch.randperm(N, generator=torch.Generator().manual_seed(42))
train_index = perm_img[:int(0.8*N)].tolist()
val_index = perm_img[int(0.8*N):].tolist()

train_dataset = datasets.CIFAR100(
    root="./data",
    train=True,
    download=False,
    transform=train_transform)

val_dataset = datasets.CIFAR100(
    root="./data",
    train=True,
    download=False,
    transform=val_transform)

test_dataset = datasets.CIFAR100(
    root="./data",
    train=False,
    download=True,
    transform=val_transform)

#DONIK: Interesantno, mozete pogledati i nacin podjele podataka s SubsetRandomSampler koji se koristi direktno u DataLoaderu

train_dataset = Subset(train_dataset, train_index)
val_dataset   = Subset(val_dataset, val_index)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)


# In[ ]:


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


# In[ ]:


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

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            prediction = outputs.argmax(dim=1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

    test_loss = test_loss / total
    test_acc = correct / total
    print(f"test_loss={test_loss:.4f} | test_acc={test_acc:.4f}")

    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(epochs, train_accs, label="Train accuracy")
    plt.plot(epochs, val_accs, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
