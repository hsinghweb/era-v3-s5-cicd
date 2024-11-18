import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

MODEL_PATH = 'mnist_model.pth'

def create_model():
    """Create the model architecture"""
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(4, 8, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        
        nn.Flatten(),
        nn.Linear(8 * 7 * 7, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 10)
    )
    return model

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def load_model():
    """Load the trained model"""
    model = create_model()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    return model

def train_model():
    """Train and save the model"""
    print("\n=== Starting Model Training ===")
    model = create_model()
    transform = get_transforms()
    
    # Setup data
    train_dataset = MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    max_epochs = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    model = model.to(device)
    
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Training phase
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        print("-" * 50)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {running_loss/100:.3f} "
                      f"Train Acc: {100.*correct_train/total_train:.2f}%")
                running_loss = 0.0
        
        # Testing phase
        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total_test += targets.size(0)
                correct_test += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct_test / total_test
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Loss: {avg_test_loss:.4f}")
        
        # Stop if we reach 95% accuracy
        if accuracy >= 95:
            print(f"\n🎉 Reached {accuracy:.2f}% accuracy. Stopping training.")
            break
    
    # Verify final accuracy meets requirement
    if accuracy < 95:
        raise ValueError(f"Model failed to achieve 95% accuracy, only reached {accuracy:.2f}%")
    
    print("\n=== Training Complete ===")
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    
    # Display parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Save the model
    torch.save(model.state_dict(), MODEL_PATH)
    return model

if __name__ == "__main__":
    model = train_model()