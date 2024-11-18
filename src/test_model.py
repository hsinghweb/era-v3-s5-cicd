import torch
from torchvision import datasets, transforms
from model import MNISTModel
import pytest
import glob
import os

def get_latest_model():
    model_files = glob.glob('models/mnist_model_*.pth')
    if not model_files:
        raise FileNotFoundError("No model files found")
    return max(model_files, key=os.path.getctime)

def test_model_architecture():
    model = MNISTModel()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape is incorrect"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
    print(f"Model has less than 25000 parameters: {total_params < 25000}")

def test_model_accuracy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    
    # Load the latest trained model
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 80, f"Model accuracy is {accuracy}%, should be > 80%"

def verify_model_requirements(model, final_accuracy):
    print("\n=== Model Requirements Check ===")
    # Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"✓ Less than 25000 parameters: {total_params < 25000}")
    
    # Check accuracy
    print(f"Training accuracy: {final_accuracy:.2f}%")
    print(f"✓ Reaches ≥95% in 1 epoch: {final_accuracy >= 95.0}")
    
    # Overall check
    requirements_met = (total_params < 25000) and (final_accuracy >= 95.0)
    print(f"\nAll requirements met: {'✓' if requirements_met else '✗'}")

if __name__ == "__main__":
    pytest.main([__file__]) 
    model = MNISTModel()
    # ... your training code ...
    
    # Calculate accuracy before verification
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct = 0
    total = 0
    
    # Reuse test accuracy calculation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    final_accuracy = 100 * correct / total
    
    # Now verify with the calculated accuracy
    verify_model_requirements(model, final_accuracy)