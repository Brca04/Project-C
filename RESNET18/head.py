import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# ----------------------
# 1. Setup
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10  # CIFAR-10
batch_size = 512
num_epochs = 20
learning_rate = 1e-3

train_losses, val_losses = [], []
train_accs, val_accs = [], []

# ----------------------
# 2. Model
# ----------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Only train classifier parameters
optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# ----------------------
# 3. Data
# ----------------------
train_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

val_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10
base_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=None)
N = len(base_dataset)
perm = torch.randperm(N, generator=torch.Generator().manual_seed(42))
train_idx, val_idx = perm[:int(0.8*N)].tolist(), perm[int(0.8*N):].tolist()

train_dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=train_transform)
val_dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=val_transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)

train_dataset = Subset(train_dataset, train_idx)
val_dataset = Subset(val_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ----------------------
# 4. Evaluation
# ----------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return val_loss / total, correct / total

# ----------------------
# 5. Training
# ----------------------
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

# ----------------------
# 6. Test
# ----------------------
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# ----------------------
# 7. Plots
# ----------------------
epochs = range(1, num_epochs+1)

plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
