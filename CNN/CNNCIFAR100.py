# =====================
# 1. IMPORTS
# =====================
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# =====================
# 2. CONFIG
# =====================
BATCH_SIZE =64
EPOCHS = 10
LR = 0.001
NUM_CLASSES = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# 3. DATASET
# =====================
def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),  # CIFAR-100 mean
                             (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),  # CIFAR-100 mean
                             (0.2675, 0.2565, 0.2761))
    ])

    train_set = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )

    test_set = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


# =====================
# 4. MODEL (SMALL CNN)
# =====================
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Linear(64 * 8 * 8, NUM_CLASSES)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# =====================
# 5. TRAIN & EVAL
# =====================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    correct, total = 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return 100 * correct / total


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100 * correct / total


# =====================
# 6. MAIN
# =====================
def main():
    train_loader, test_loader = get_dataloaders()

    model = SmallCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_acc = evaluate(model, test_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")


# =====================
# 7. RUN
# =====================
if __name__ == "__main__":
    main()
