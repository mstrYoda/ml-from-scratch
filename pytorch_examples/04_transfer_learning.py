import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models

# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 10 classes

# Define transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Example of using the model
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print("Output shape:", output.shape)

# Training setup (example)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

print("Model ready for fine-tuning") 