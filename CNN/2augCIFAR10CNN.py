#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Subset


# In[2]:


class smallCN(nn.Module):
    def __init__(self):
        super(smallCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2)
        #for CIFAR10 32x32
        self.fc = nn.Linear(64*8*8, 10)
        #DONIK: Pogledajte i kako bi implementirali mrezu koja je potpuno konvolucijska, ne morate implementirati (hint: Global Average Pooling)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.relu(x)

        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.relu(x)

        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smallCN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# In[3]:


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )#ovdje nadodavati augmentacije
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )#nadodati std i mean
])

base_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=None
)

N = len(base_dataset)
perm_img = torch.randperm(N, generator=torch.Generator().manual_seed(42))
train_index = perm_img[:int(0.8*N)].tolist()
val_index = perm_img[int(0.8*N):].tolist()

train_dataset = datasets.CIFAR10(
    root="./data", 
    train=True, 
    download=False, 
    transform=train_transform)

val_dataset = datasets.CIFAR10(
    root="./data", 
    train=True, 
    download=False, 
    transform=val_transform)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

train_dataset = Subset(train_dataset, train_index)
val_dataset   = Subset(val_dataset, val_index)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[ ]:


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            prediction = outputs.argmax(dim=1)
            val_correct += (prediction == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    return val_loss, val_acc



# In[4]:


epohs = 20

for i in range(epohs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        prediction = outputs.argmax(dim=1)
        correct += (prediction == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    print(f"Epoch {i+1}/{epohs} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f}")

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {i+1}/{epohs} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

#DONIK: Predlazem da si pospremite sve lossove (i train i val) da mozete kasnije plotati grafove.


# In[ ]:


model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        prediction = outputs.argmax(dim=1)
        test_correct += (prediction == labels).sum().item()
        test_total += labels.size(0)

test_loss = test_loss / test_total
test_acc = test_correct / test_total
print(f"test_loss={test_loss:.4f} | test_acc={test_acc:.4f}")    

