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

@pytest.fixture(scope="session")
def trained_model():
    """Create or load model for all tests"""
    if not Path(MODEL_PATH).exists():
        model = train_model()
    else:
        model = load_model()
    return model

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

def test_model_requirements(trained_model):
    """Test model size and verify it's trained"""
    assert trained_model is not None, "Model was not properly loaded"
    num_params = sum(p.numel() for p in trained_model.parameters())
    assert num_params <= 25000, f"Model has {num_params:,} parameters (must be ≤ 25,000)"
    
    # Verify model is trained (weights aren't random)
    sample_weight = next(trained_model.parameters())
    assert not torch.allclose(sample_weight, torch.zeros_like(sample_weight)), "Model appears untrained"

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