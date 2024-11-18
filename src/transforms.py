from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image

def get_rotation_transform(degree):
    return transforms.Compose([
        transforms.RandomRotation([degree, degree]),  # Fixed degree rotation
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def show_rotated_samples():
    # Load one image from training dataset
    dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=None)
    original_img, _ = dataset[0]  # Get the first image
    
    # Create subplot with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('MNIST Digit with Different Rotations', fontsize=16)
    
    # Show original
    original_tensor = transforms.ToTensor()(original_img)
    axes[0][0].imshow(original_tensor.squeeze(), cmap='gray')
    axes[0][0].set_title('Original')
    axes[0][0].axis('off')
    
    # Show different rotations
    rotation_degrees = [15, 30, 45, -30, -15]
    positions = [(0,1), (0,2), (1,0), (1,1), (1,2)]  # Fixed positions for each rotation
    
    for (degree, pos) in zip(rotation_degrees, positions):
        # Apply rotation transform to PIL Image
        rotation_transform = get_rotation_transform(degree)
        rotated_img = rotation_transform(original_img)
        
        row, col = pos
        axes[row][col].imshow(rotated_img.squeeze(), cmap='gray')
        axes[row][col].set_title(f'Rotation: {degree}°')
        axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_rotated_samples() 