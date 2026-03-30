# ======================================
# RESNET18 ON CIFAR-10 (FULL SCRIPT)
# ======================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ======================================
# CONFIG
# ======================================
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0005
NUM_CLASSES = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================
# DATASET
# ======================================
def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    train_set = torchvision.datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    test_set = torchvision.datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


# ======================================
# MODEL
# ======================================
def get_model():
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )

    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


# ======================================
# TRAINING
# ======================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    correct, total, running_loss = 0, 0, 0.0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    loss = running_loss / len(loader)

    return acc, loss


# ======================================
# EVALUATION
# ======================================
def evaluate(model, loader, criterion):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    loss = running_loss / len(loader)

    return acc, loss


# ======================================
# MAIN
# ======================================
def main():
    print(f"Using device: {DEVICE}")

    train_loader, test_loader = get_dataloaders()

    model = get_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        train_acc, train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        test_acc, test_loss = evaluate(
            model, test_loader, criterion
        )

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_resnet18_cifar10.pth")

    print(f"Best Test Accuracy: {best_acc:.2f}%")


# ======================================
# RUN
# ======================================
if __name__ == "__main__":
    main()
