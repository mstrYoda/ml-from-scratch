import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Custom dataset that uses both NumPy and PyTorch
class CustomDataset(Dataset):
    def __init__(self, size=1000):
        # Generate data using NumPy
        self.data = np.random.randn(size, 10)
        self.labels = np.random.randint(0, 2, size)
        
        # Convert to PyTorch
        self.data = torch.FloatTensor(self.data)
        self.labels = torch.LongTensor(self.labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create dataset and dataloader
dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example of processing pipeline
def process_batch(batch_data, batch_labels):
    # NumPy preprocessing
    numpy_data = batch_data.numpy()
    normalized = (numpy_data - np.mean(numpy_data, axis=0)) / np.std(numpy_data, axis=0)
    
    # Back to PyTorch
    torch_normalized = torch.from_numpy(normalized)
    return torch_normalized, batch_labels

# Training loop example
print("Processing batches:")
for i, (data, labels) in enumerate(dataloader):
    processed_data, processed_labels = process_batch(data, labels)
    print(f"Batch {i+1}: Data shape: {processed_data.shape}, Labels shape: {processed_labels.shape}")
    if i == 2:  # Show only first 3 batches
        break 