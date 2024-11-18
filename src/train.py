import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from model import MNISTModel

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)

    # Training
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Accumulate running loss
        running_loss += loss.item()
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            print(f'Batch [{batch_idx:>4d}/{len(train_loader)}] '
                  f'({100. * batch_idx / len(train_loader):>3.0f}%) | '
                  f'Loss: {loss.item():.4f} | '
                  f'Avg Loss: {avg_loss:.4f}')

    # Print final epoch statistics
    final_avg_loss = running_loss / len(train_loader)
    print(f'\nTraining completed. Final average loss: {final_avg_loss:.4f}')

    return model 