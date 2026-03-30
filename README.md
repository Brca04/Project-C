# Image-Classification-Using-Neural-Networks
Project R from the course [Project R](https://www.fer.unizg.hr/en/course/proc) at FER, academic year 2025/2026.

## About
Implementation and comparison of image classification methods based on convolutional neural networks (CNNs). The project explores the training process of various CNN architectures and compares different classification approaches across multiple benchmark datasets.

## Neural Network Architectures
- **Small CNN** — a minimal convolutional network with a few layers
- **ResNet-18** — residual network with skip connections solving vanishing/exploding gradient problems
- **VGG-16** — 16-layer network using stacked 3×3 convolutions
- **EfficientNet-B0** — baseline of the EfficientNet family using compound scaling
- **EfficientNet-B1** — scaled-up version with MBConv blocks and Squeeze-and-Excitation
- **ConvNeXt** — modernized CNN inspired by Transformer design principles

## Classification Methods
- Training from scratch
- Fine-tuning (pretrained ImageNet1k weights)
- Classification head training (frozen backbone)
- Linear probe
- Support Vector Machine (SVM)
- k-Nearest Neighbours (kNN)

## Datasets
- **CIFAR-10** — 10 classes
- **CIFAR-100** — 100 classes
- **Imagenette** — 10-class subset of ImageNet

## Key Results
| Model | CIFAR-10 | CIFAR-100 | Imagenette |
|---|---|---|---|
| Small CNN | 0.7831 (scratch) | 0.5169 (scratch) | 0.6245 (scratch) |
| ResNet-18 | 0.9467 (scratch) | 0.7852 (fine-tuning) | 0.9791 (fine-tuning) |
| VGG-16 | 0.8400 (linear probe) | 0.6884 (class. head) | 0.9684 (SVM) |
| EfficientNet-B0 | 0.9116 (SVM) | 0.7107 (SVM) | 0.9801 (fine-tuning) |
| EfficientNet-B1 | 0.9126 (SVM) | 0.8200 (class. head) | 0.9819 (SVM) |
| ConvNeXt | 0.9411 (linear probe) | 0.8113 (linear probe) | 0.9977 (SVM) |

*Best test accuracy per model and dataset. Best method shown in parentheses.*

ConvNeXt consistently achieved the highest accuracy across most experiments, while SVM and kNN proved to be the most effective classification methods overall.

## Team
Luka Čečura, Lucija Kozić, Leona Križanac, Nora Milolović, Vita Pavlović, Pavle Stanarević, Nika Valić, Daniel Žic and Bruno Cavor (team lead)

**Mentor:** Akademik prof. dr. sc. Sven Lončarić

## Note
**Data used for training was not uploaded to repository (will be downloaded automatically when running code).**
