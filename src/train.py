import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from model import MNISTModel
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Define augmentation transforms
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def show_augmented_samples(dataset, num_samples=5):
    plt.figure(figsize=(12, 3))
    for i in range(num_samples):
        # Get the same image multiple times to show different augmentations
        img, label = dataset[0]
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')
    plt.savefig('augmented_samples.png')
    plt.close()

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),     # Reduced to 4 filters
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(4, 8, 3, padding=1),     # Reduced to 8 filters
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        
        nn.Flatten(),
        nn.Linear(8 * 7 * 7, 32),          # Reduced input channels to 8
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 10)
    )

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nTotal parameters: {total_params:,}')

    optimizer = optim.Adam(model.parameters(), lr=0.002)  # Slightly higher learning rate
    criterion = nn.CrossEntropyLoss()
    
    # Simplified augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform = get_transforms()
    train_dataset = MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Single epoch training with detailed logging
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            current_acc = 100 * correct / total
            print(f'Batch [{batch_idx:>4d}/{len(train_loader)}] '
                  f'({100. * batch_idx / len(train_loader):>3.0f}%) | '
                  f'Loss: {loss.item():.4f} | '
                  f'Avg Loss: {avg_loss:.4f} | '
                  f'Acc: {current_acc:.2f}%')

    final_acc = 100 * correct / total
    print(f'\nTraining completed. Final accuracy: {final_acc:.2f}%')
    return model 

if __name__ == "__main__":
    # Train the model
    model = train_model()
    
    # Create dataset
    transform = get_transforms()
    train_dataset = MNIST('./data', train=True, download=True, transform=transform)
    
    # Show samples (only when running locally, not in CI)
    import os
    if not os.environ.get('CI'):  # Skip visualization in CI environment
        show_augmented_samples(train_dataset)