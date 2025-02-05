import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# Create a dummy image using NumPy
def create_sample_image():
    # Create a gradient image using NumPy
    x = np.linspace(0, 1, 224)
    y = np.linspace(0, 1, 224)
    xx, yy = np.meshgrid(x, y)
    image = np.stack([xx, yy, xx], axis=2)
    return (image * 255).astype(np.uint8)

# Image processing pipeline
def process_image():
    # Create image with NumPy
    numpy_image = create_sample_image()
    print("NumPy image shape:", numpy_image.shape)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(numpy_image)
    
    # Define PyTorch transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to PyTorch tensor
    torch_image = transform(pil_image)
    print("PyTorch tensor shape:", torch_image.shape)
    
    # Back to NumPy for additional processing
    numpy_processed = torch_image.numpy()
    print("Processed NumPy shape:", numpy_processed.shape)
    
    return numpy_processed

# Demonstrate the pipeline
processed_image = process_image()

# Batch processing example
def batch_process_images(batch_size=4):
    images = []
    for _ in range(batch_size):
        numpy_image = create_sample_image()
        images.append(numpy_image)
    
    # Stack with NumPy
    numpy_batch = np.stack(images)
    print("\nNumPy batch shape:", numpy_batch.shape)
    
    # Convert to PyTorch
    torch_batch = torch.from_numpy(numpy_batch).float().permute(0, 3, 1, 2)
    print("PyTorch batch shape:", torch_batch.shape)
    
    return torch_batch

# Run batch processing
batch_images = batch_process_images() 