# Datathon
# Meme Classification Model - Step by Step Guide
# This model will classify memes into: Politics, Gender, Religion, or Neutral

# ============================================
# STEP 1: Install Required Libraries
# ============================================
# Run these in terminal before starting:
# pip install tensorflow keras pillow pandas numpy matplotlib scikit-learn

# ============================================
# STEP 2: Import Libraries
# ============================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================================
# STEP 3: Load and Prepare Your Data
# ============================================
# Assuming your data structure is:
# data/
#   ├── image1.jpg (with label in CSV or folder name)
#   ├── image2.jpg
#   └── ...
# labels.csv with columns: [filename, category]

def load_data(image_folder, labels_csv):
    """
    Load images and their corresponding labels
    """
    df = pd.read_csv(labels_csv)
    
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        img_path = os.path.join(image_folder, row['filename'])
        
        try:
            # Read and resize image to 224x224 (standard size)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))
            
            # Convert image to array
            img_array = np.array(img) / 255.0  # Normalize to 0-1
            
            images.append(img_array)
            labels.append(row['category'])
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    return np.array(images), np.array(labels)

# Load your data
print("Loading images and labels...")
image_folder = "path/to/your/images"  # Change this path
labels_csv = "path/to/labels.csv"      # Change this path

X, y = load_data(image_folder, labels_csv)
print(f"Loaded {len(X)} images")
print(f"Categories: {np.unique(y)}")

# ============================================
# STEP 4: Encode Labels (Convert text to numbers)
# ============================================
# Politics -> 0, Gender -> 1, Religion -> 2, Neutral -> 3
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print(f"Label mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

# ============================================
# STEP 5: Split Data into Training and Testing
# ============================================
# 80% for training (to teach the model)
# 20% for testing (to check how well it works)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2,        # 20% for testing
    random_state=42,
    stratify=y_encoded    # Keep class distribution same
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ============================================
# STEP 6: Build the Neural Network Model
# ============================================
# Think of it like stacking layers of filters to understand images better

model = keras.Sequential([
    # Input layer
    layers.Input(shape=(224, 224, 3)),
    
    # Convolutional layers (learn image patterns)
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten: convert 2D image data to 1D
    layers.Flatten(),
    
    # Dense layers (make final decisions)
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  # Prevent overfitting
    
    # Output layer (4 categories)
    layers.Dense(4, activation='softmax')
])

# Print model structure
model.summary()

# ============================================
# STEP 7: Compile the Model
# ============================================
# Tell the model how to learn and what to measure
model.compile(
    optimizer='adam',                    # Learning algorithm
    loss='sparse_categorical_crossentropy',  # For multi-class classification
    metrics=['accuracy']                 # What to measure
)

# ============================================
# STEP 8: Train the Model
# ============================================
# This teaches the model to recognize meme categories
history = model.fit(
    X_train, y_train,
    epochs=15,              # Number of times to go through all training data
    batch_size=32,          # Process 32 images at a time
    validation_split=0.2,   # Use 20% of training data to validate
    verbose=1
)

# ============================================
# STEP 9: Evaluate on Test Data
# ============================================
# Check how well the model performs on unseen data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# ============================================
# STEP 10: Make Predictions
# ============================================
def predict_meme_category(image_path, model, encoder):
    """
    Predict the category of a single meme
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    category = encoder.inverse_transform([predicted_class])[0]
    
    return category, confidence

# Example: Predict a new meme
# test_image = "path/to/test/meme.jpg"
# category, confidence = predict_meme_category(test_image, model, encoder)
# print(f"Category: {category} (Confidence: {confidence * 100:.2f}%)")

# ============================================
# STEP 11: Save Your Model
# ============================================
model.save('meme_classifier_model.h5')
print("Model saved as 'meme_classifier_model.h5'")

# ============================================
# STEP 12: Visualize Training Progress
# ============================================
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

print("Training visualization saved as 'training_history.png'")


# ============================================
# METHOD 2: Using CSV with Image Paths (Most Common)
# ============================================
# Your CSV looks like:
# image_path,category
# images/meme1.jpg,Politics
# images/meme2.jpg,Neutral
# images/meme3.jpg,Gender

def load_images_from_csv(csv_file, base_path='', img_size=(224, 224)):
    """
    Load images using a CSV file with image paths and labels
    """
    df = pd.read_csv(csv_file)
    
    images = []
    labels = []
    
    print(f"Total images in CSV: {len(df)}")
    
    for idx, row in df.iterrows():
        # Show progress
        if (idx + 1) % 100 == 0:
            print(f"Loaded {idx + 1}/{len(df)} images...")
        
        # Construct full path
        img_path = os.path.join(base_path, row['image_path'])
        category = row['category']
        
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Resize
            img = img.resize(img_size)
            
            # Normalize (convert to values between 0 and 1)
            img_array = np.array(img) / 255.0
            
            images.append(img_array)
            labels.append(category)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    return np.array(images), np.array(labels)

# Usage:
X, y = load_images_from_csv('labels.csv', base_path='.')
print(f"Loaded {len(X)} images")


# ============================================
# METHOD 3: Using Keras Image Generator (Best for Large Datasets)
# ============================================
# This is memory efficient - loads images in batches during training
# File structure same as METHOD 1

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_with_image_generator(train_folder, img_size=(224, 224)):
    """
    Use ImageDataGenerator for efficient loading
    Great for large datasets that don't fit in memory
    """
    # Create data generator with normalization
    datagen = ImageDataGenerator(
        rescale=1./255,           # Normalize
        rotation_range=20,        # Random rotation
        width_shift_range=0.2,    # Random horizontal shift
        height_shift_range=0.2,   # Random vertical shift
        shear_range=0.2,          # Shear transformation
        zoom_range=0.2,           # Random zoom
        horizontal_flip=True,     # Random flip
        fill_mode='nearest'
    )
    
    # Load images from directory
    train_generator = datagen.flow_from_directory(
        train_folder,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical'  # For multiple classes
    )
    
    return train_generator

# Usage:
train_gen = load_with_image_generator('data')
# This returns a generator, not actual data
# Use it directly in model.fit()
