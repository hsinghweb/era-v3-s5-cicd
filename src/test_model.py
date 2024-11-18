import torch
import pytest
from train import train_model
from torchvision import datasets, transforms

def verify_model_requirements(model, accuracy):
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Check requirements
    param_check = total_params <= 25000
    accuracy_check = accuracy >= 95.0
    
    # Print results
    print("\n=== Model Requirements Check ===")
    print(f"1. Parameter count ({total_params:,}): {'✓' if param_check else '✗'}")
    print(f"2. Accuracy check ({accuracy:.2f}%): {'✓' if accuracy_check else '✗'}")
    
    # Assert both conditions
    assert param_check, f"Model has {total_params:,} parameters (must be ≤ 25,000)"
    assert accuracy_check, f"Model accuracy ({accuracy:.2f}%) is below required 95%"

@pytest.mark.filterwarnings("ignore")
def test_model_requirements(capsys):
    print("\n=== Starting Model Training and Testing ===")
    print("Training model...")
    model = train_model()
    
    print("\nCalculating final accuracy...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct = 0
    total = 0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            if batch_idx % 10 == 0:
                print(f"Progress: {batch_idx * 1000}/{len(train_dataset)} images processed")
    
    final_accuracy = 100 * correct / total
    verify_model_requirements(model, final_accuracy)

if __name__ == "__main__":
    model = train_model()
    
    # Calculate training accuracy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    train_correct = 0
    train_total = 0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000)
    
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
    
    final_accuracy = 100 * train_correct / train_total
    verify_model_requirements(model, final_accuracy)