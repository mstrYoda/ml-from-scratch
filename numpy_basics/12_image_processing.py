import numpy as np
from matplotlib import pyplot as plt

# Create a simple image
image = np.zeros((10, 10))
image[2:8, 2:8] = 1
print("Simple binary image:\n", image)

# Image transformations
rotated = np.rot90(image)
flipped = np.fliplr(image)

# Display images
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.subplot(132)
plt.imshow(rotated, cmap='gray')
plt.title('Rotated')
plt.subplot(133)
plt.imshow(flipped, cmap='gray')
plt.title('Flipped')
plt.show()

# Image filtering
noisy = image + np.random.normal(0, 0.2, image.shape)
filtered = np.clip(noisy, 0, 1)  # Clip values between 0 and 1 