import torch
import pytest
from train import train_model, get_transforms
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from pytest import ExitCode

@pytest.fixture(scope="session")
def trained_model():
    """Create model once and reuse for all tests"""
    return train_model()

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
def test_model_requirements(trained_model):
    print("\nCalculating final accuracy...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.eval()
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
            outputs = trained_model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            if batch_idx % 10 == 0:
                print(f"Progress: {batch_idx * 1000}/{len(train_dataset)} images processed")
    
    final_accuracy = 100 * correct / total
    verify_model_requirements(trained_model, final_accuracy)

def test_model_output_shape(trained_model):
    """Test if model outputs correct shape (batch_size, 10) for MNIST"""
    print("\n=== Test if model outputs correct shape (batch_size, 10) for MNIST ===")
    batch_size = 64
    dummy_input = torch.randn(batch_size, 1, 28, 28)  # MNIST image size
    output = trained_model(dummy_input)
    assert output.shape == (batch_size, 10), f"Expected shape (64, 10), got {output.shape}"

def test_transform_normalization():
    """Test if transforms normalize images to expected range"""
    print("\n=== Test if transforms normalize images to expected range ===")
    transform = get_transforms()
    # Create a dummy PIL image (gray)
    dummy_image = Image.fromarray(np.uint8(np.ones((28, 28)) * 128))
    transformed = transform(dummy_image)
    
    # For MNIST normalization (mean=0.1307, std=0.3081)
    # A gray image (128/255 ≈ 0.5) should be transformed to approximately:
    # (0.5 - 0.1307) / 0.3081 ≈ 1.2
    assert -2 < transformed.mean() < 2, f"Transform normalization not in expected range: {transformed.mean()}"

def test_model_training_mode(trained_model):
    """Test if dropout layers behave differently in train vs eval mode"""
    print("\n=== Test if dropout layers behave differently in train vs eval mode ===")
    # Same input in train mode
    trained_model.train()
    input_tensor = torch.randn(1, 1, 28, 28)
    
    # Run multiple forward passes and collect outputs
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
    assert torch.allclose(eval_output1, eval_output2), "Outputs should be identical in eval mode"

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