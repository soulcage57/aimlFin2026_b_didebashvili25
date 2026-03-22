
# Convolutional Neural Networks (CNNs)
### 1. Introduction to CNNs 
A Convolutional Neural Network (CNN) is a class of deep neural networks most commonly applied to analyzing visual imagery. While traditional Multi-Layer Perceptrons (MLPs) struggle with the high dimensionality of images (where every pixel is a feature), CNNs use a "sliding window" approach to capture spatial hierarchies and local patterns.

**Core Architecture** \
A standard CNN consists of three main types of layers: \
**Convolutional Layers:** These are the building blocks. They use filters (or kernels) that slide across the input data to perform element-wise multiplication. This process produces a Feature Map, highlighting specific traits like edges, textures, or shapes \
**Pooling Layers:** To reduce the computational load and prevent overfitting, pooling layers downsample the feature maps. The most common type is Max Pooling, which takes the maximum value from a specific window. \
**Fully Connected** (FC) Layers: After several rounds of convolution and pooling, the flattened output is fed into a traditional neural network layer to perform the final classification.
<img width="1617" height="785" alt="The_Architecture_of_Convolutional_Neural_Networks_8263469ad1" src="https://github.com/user-attachments/assets/050b6ec0-c800-481a-802e-97edaaca081d" />


### 2. Mathematical Foundation
The core operation, the convolution, can be expressed mathematically. Given an input image $I$ and a kernel $K$, the output feature map $S(i, j)$ is calculated as: 
$$S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i + m, j + n) K(m, n)$$  \
This operation allows the network to achieve translation invariance, meaning it can recognize a pattern regardless of where it appears in the frame

### 3. Practical Example: Malware Detection from Binary Data

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

**Sample dataset**
X = np.array([
    [34, 87, 123],
    [12, 45, 67],
    [90, 210, 33],
    [56, 78, 90]
])

y = np.array([1, 0, 1, 0])

**Reshape for CNN (samples, height, width, channels)**
X = X.reshape((4, 3, 1, 1))

**Build CNN model**
model = models.Sequential([
    layers.Conv2D(16, (2,1), activation='relu', input_shape=(3,1,1)),
    layers.MaxPooling2D((1,1)),
    layers.Flatten(),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

**Train model**
model.fit(X, y, epochs=10, verbose=1)

**Predict**
predictions = model.predict(X)
print(predictions)
```

## Training Accuracy
<img width="1309" height="803" alt="image" src="https://github.com/user-attachments/assets/6692bca1-79f4-4cce-bc61-c63456aadf04" /> 


## Predictions
<img width="1342" height="776" alt="image" src="https://github.com/user-attachments/assets/18b9b70a-3b15-4082-b5b5-ec49b0b27b98" />










>>>>>>> 2e83b7913f56aa0abb5524d01503793d0718f5be
