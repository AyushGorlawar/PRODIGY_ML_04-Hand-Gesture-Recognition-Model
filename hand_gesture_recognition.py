import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMG_SIZE = 64  # Define the image size for resizing

def load_data(data_dir, img_size=IMG_SIZE):
    images = []
    labels = []
    for idx, folder in enumerate(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        print(f"Processing folder: {folder_path}")  # Debug statement
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file_path.endswith('.png'):
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (img_size, img_size))
                    images.append(img)
                    labels.append(idx)
    print(f"Total images: {len(images)}")  # Debug statement
    print(f"Total labels: {len(labels)}")  # Debug statement
    images = np.array(images).reshape(-1, img_size, img_size, 1) / 255.0
    labels = to_categorical(labels)
    return images, labels

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Define the data directory
    data_dir = 'data/leapGestRecog'

    # Load and preprocess the data
    X, y = load_data(data_dir)
    if len(X) == 0 or len(y) == 0:
        print("Error: No data loaded. Check the dataset directory and file paths.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the model
    model = create_model((IMG_SIZE, IMG_SIZE, 1), y_train.shape[1])

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model's performance
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy*100:.2f}%')

    # Save the model
    model.save('gesture_model.h5')

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

if __name__ == "__main__":
    main()
