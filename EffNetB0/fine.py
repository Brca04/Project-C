# =========================
# Imports
# =========================
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# Hyperparameters
# =========================
num_classes = 100
batch_size = 512
epochs = 20
lr = 1e-3
weight_decay = 1e-4

# =========================
# Model (Pretrained)
# =========================
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(device)

# =========================
# Loss & Optimizer
# =========================
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

# =========================
# Transforms (CIFAR-100)
# =========================
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
])

# =========================
# Dataset & Split
# =========================
base_dataset = datasets.CIFAR100(
    root="./data",
    train=True,
    download=True
)

N = len(base_dataset)
perm = torch.randperm(N, generator=torch.Generator().manual_seed(42))
train_idx = perm[:int(0.8 * N)]
val_idx = perm[int(0.8 * N):]

train_dataset = Subset(
    datasets.CIFAR100("./data", train=True, transform=train_transform),
    train_idx
)

val_dataset = Subset(
    datasets.CIFAR100("./data", train=True, transform=val_transform),
    val_idx
)

test_dataset = datasets.CIFAR100(
    "./data",
    train=False,
    download=True,
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =========================
# Evaluation function
# =========================
def evaluate(model, loader):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return loss_sum / total, correct / total

# =========================
# Training Loop
# =========================
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(epochs):
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
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    val_loss, val_acc = evaluate(model, val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train_loss={train_loss:.4f} | Train_acc={train_acc:.4f} | "
        f"Val_loss={val_loss:.4f} | Val_acc={val_acc:.4f}"
    )

# =========================
# Test Evaluation
# =========================
test_loss, test_acc = evaluate(model, test_loader)
print(f"\nTest_loss={test_loss:.4f} | Test_acc={test_acc:.4f}")

# =========================
# Plots
# =========================
epochs_range = range(1, epochs + 1)

plt.figure()
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(epochs_range, train_accs, label="Train Acc")
plt.plot(epochs_range, val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
