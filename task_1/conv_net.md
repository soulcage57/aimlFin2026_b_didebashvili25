import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# --------------------------------------------------
# 1. Convert binary file to grayscale image
# --------------------------------------------------
def binary_to_image(file_path, image_size=(64, 64)):
    """
    Converts a binary file into a grayscale image.
    Each byte (0–255) becomes a pixel value.
    """
    with open(file_path, 'rb') as f:
        byte_data = f.read(image_size[0] * image_size[1])
    
    # Pad if file is too small
    if len(byte_data) < image_size[0] * image_size[1]:
        byte_data += b'\x00' * (image_size[0] * image_size[1] - len(byte_data))
    
    image = np.frombuffer(byte_data, dtype=np.uint8)
    return image.reshape(image_size)


# --------------------------------------------------
# 2. Load dataset from folders
# --------------------------------------------------
def load_dataset(malware_dir, benign_dir, image_size=(64, 64)):
    """
    Loads malware and benign files, converts them into images,
    and assigns labels.
    """
    X = []
    y = []

    # Malware samples (label = 1)
    for file in os.listdir(malware_dir):
        path = os.path.join(malware_dir, file)
        try:
            img = binary_to_image(path, image_size)
            X.append(img)
            y.append(1)
        except:
            continue

    # Benign samples (label = 0)
    for file in os.listdir(benign_dir):
        path = os.path.join(benign_dir, file)
        try:
            img = binary_to_image(path, image_size)
            X.append(img)
            y.append(0)
        except:
            continue

    X = np.array(X).reshape(-1, image_size[0], image_size[1], 1) / 255.0
    y = np.array(y)

    return X, y


# --------------------------------------------------
# 3. Define CNN Model
# --------------------------------------------------
def create_model(input_shape=(64, 64, 1)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# --------------------------------------------------
# 4. Example Usage
# --------------------------------------------------
if __name__ == "__main__":
    # Replace with your actual dataset paths
    malware_path = "data/malware/"
    benign_path = "data/benign/"

    print("Loading dataset...")
    X, y = load_dataset(malware_path, benign_path)

    print(f"Dataset loaded: {X.shape[0]} samples")

    model = create_model()

    print("Training model...")
    model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

    print("Training complete.")