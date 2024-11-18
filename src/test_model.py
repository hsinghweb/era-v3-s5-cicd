import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from train import (
    train_model, 
    load_model, 
    MODEL_PATH, 
    get_transforms
)
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

@pytest.fixture(scope="session")
def trained_model():
    """Create or load model for all tests"""
    if not Path(MODEL_PATH).exists():
        model = train_model()
    else:
        model = load_model()
    return model

def test_model_requirements(trained_model):
    """Test model size and accuracy requirements"""
    print("\n=== Testing Model Requirements ===")
    
    # Test parameter count
    num_params = sum(p.numel() for p in trained_model.parameters())
    print(f"Total parameters: {num_params:,}")
    assert num_params <= 25000, f"Model has {num_params:,} parameters (must be ≤ 25,000)"
    
    # Test accuracy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.eval()
    test_dataset = MNIST('./data', train=False, transform=get_transforms())
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = trained_model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    print(f"Model accuracy: {accuracy:.2f}%")
    assert accuracy >= 95, f"Model accuracy {accuracy:.2f}% is below required 95%"
    
    print("✅ Model meets all requirements:")
    print(f"  • Parameters: {num_params:,} ≤ 25,000")
    print(f"  • Accuracy: {accuracy:.2f}% ≥ 95%")

def test_model_output_shape(trained_model):
    """Test if model outputs correct shape"""
    assert trained_model is not None, "Model was not properly loaded"
    batch_size = 64
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    output = trained_model(dummy_input)
    assert output.shape == (batch_size, 10)

def test_transform_normalization():
    """Test if transforms normalize images to expected range"""
    transform = get_transforms()
    # Create a dummy PIL image (gray)
    dummy_image = Image.fromarray(np.uint8(np.ones((28, 28)) * 128))
    transformed = transform(dummy_image)
    
    # For MNIST normalization (mean=0.1307, std=0.3081)
    assert -2 < transformed.mean() < 2, f"Transform normalization not in expected range: {transformed.mean()}"

def test_model_training_mode(trained_model):
    """Test if dropout layers behave differently in train vs eval mode"""
    assert trained_model is not None, "Model was not properly loaded"
    input_tensor = torch.randn(1, 1, 28, 28)
    
    # Test train mode
    trained_model.train()
    outputs = []
    for _ in range(5):
        outputs.append(trained_model(input_tensor).detach())
    
    # Check if at least one pair of outputs is different
    all_same = all(torch.allclose(outputs[0], output) for output in outputs[1:])
    assert not all_same, "Dropout doesn't seem to be working in training mode"
    
    # Test eval mode
    trained_model.eval()
    eval_output1 = trained_model(input_tensor)
    eval_output2 = trained_model(input_tensor)
    assert torch.allclose(eval_output1, eval_output2)

if __name__ == "__main__":
    # Run tests and store results
    results = pytest.main([__file__, "-v"])
    
    # Print custom summary
    print("\n=== Test Execution Summary ===")
    test_functions = [
        'test_model_requirements',
        'test_model_output_shape',
        'test_transform_normalization',
        'test_model_training_mode'
    ]
    
    for test in test_functions:
        result = "PASSED" if results == 0 else "FAILED"
        print(f"{test}: {result}")