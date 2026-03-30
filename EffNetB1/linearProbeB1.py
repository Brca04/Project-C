import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#################################
# DEVICE
#################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#################################
# DATASET
#################################

data_dir = r"C:\Users\Bruno\Desktop\FER\PROJEKT R\CNN\data\imagenette2\imagenette2"

# EfficientNet-B1 prefers 240x240 images
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(240),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(240),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])

train_dataset = datasets.ImageFolder(
    root=f"{data_dir}/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=f"{data_dir}/val",
    transform=val_transform
)

# 🔥 Adjust batch size according to your VRAM
batch_size = 1024

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

#################################
# MODEL — LINEAR PROBE (EffNet-B1)
#################################

weights = models.EfficientNet_B1_Weights.DEFAULT
model = models.efficientnet_b1(weights=weights)

# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

num_classes = 10
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model = model.to(device)

#################################
# OPTIMIZER & LOSS
#################################

optimiser = torch.optim.Adam(
    model.classifier.parameters(),
    lr=7e-4
)

criterion = nn.CrossEntropyLoss()

#################################
# MIXED PRECISION
#################################

scaler = torch.amp.GradScaler(device='cuda')

#################################
# TRACKING
#################################

train_losses = []
val_losses = []
train_accs = []
val_accs = []

#################################
# EVALUATION
#################################

def evaluate(model, loader):
    model.eval()
    loss_total = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss_total += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return loss_total / total, correct / total

#################################
# TRAINING LOOP
#################################

epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiser.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    val_loss, val_acc = evaluate(model, val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f}")
    print(f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

#################################
# FINAL TEST (Imagenette uses val)
#################################

test_loss, test_acc = evaluate(model, val_loader)
print(f"\ntest_loss={test_loss:.4f} | test_acc={test_acc:.4f}")

#################################
# PLOTS
#################################

epochs_range = range(1, epochs + 1)

plt.figure()
plt.plot(epochs_range, train_losses, label="Train loss")
plt.plot(epochs_range, val_losses, label="Validation loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(epochs_range, train_accs, label="Train accuracy")
plt.plot(epochs_range, val_accs, label="Validation accuracy")
plt.legend()
plt.grid(True)
plt.show()
