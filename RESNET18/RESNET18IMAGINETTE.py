# =====================================================
# RESNET18 ON IMAGENETTE (Python 3.12 compatible)
# =====================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


# =====================================================
# CONFIGURATION
# =====================================================
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.0005
NUM_CLASSES = 10   # Imagenette has 10 classes
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ⚠️ CHANGE THIS PATH
IMAGENETTE_ROOT = r"C:\Users\Bruno\Desktop\FER\PROJEKT R\CNN\data\imagenette2\imagenette2" 
# Folder must contain: train/ and val/


# =====================================================
# DATASET
# =====================================================
def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    train_dataset = ImageFolder(
        root=f"{IMAGENETTE_ROOT}/train",
        transform=transform_train
    )

    val_dataset = ImageFolder(
        root=f"{IMAGENETTE_ROOT}/val",
        transform=transform_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader


# =====================================================
# MODEL
# =====================================================
def get_model():
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )

    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


# =====================================================
# TRAINING
# =====================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    loss = running_loss / len(loader)

    return acc, loss


# =====================================================
# EVALUATION
# =====================================================
def evaluate(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    loss = running_loss / len(loader)

    return acc, loss


# =====================================================
# MAIN
# =====================================================
def main():
    print(f"Using device: {DEVICE}")

    train_loader, val_loader = get_dataloaders()

    model = get_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        train_acc, train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        val_acc, val_loss = evaluate(
            model, val_loader, criterion
        )

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_resnet18_imagenette.pth")

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")


# =====================================================
# RUN SCRIPT
# =====================================================
if __name__ == "__main__":
    main()
