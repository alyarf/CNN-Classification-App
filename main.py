# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Dezarhiveaza dataset-ul
data_dir = './headgear'
!unzip headgear-image-classification.zip -d {data_dir}

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

train_dataset = image_dataset_from_directory(
    f"{data_dir}/train",
    image_size=(224, 224),
    batch_size=128,
    shuffle=True
)

val_dataset = image_dataset_from_directory(
    f"{data_dir}/train",
    image_size=(224, 224),
    shuffle=True
)

test_dataset = image_dataset_from_directory(
    f"{data_dir}/test",     # folosim setul original de test
    image_size=(224, 224),
    batch_size=128,
    shuffle=False
)

class_names = train_dataset.class_names

# Normalizare pentru toate seturile de date
train_dataset = train_dataset.map(normalize_img)
val_dataset = val_dataset.map(normalize_img)
test_dataset = test_dataset.map(normalize_img)

# Augmentare date doar pentru training
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))

# Crearea modelului CNN
def generate_model():
    model = tf.keras.Sequential([
        # Primul strat convolutional
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        # Al doilea strat convolutional
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        # Strat complet conectat
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])
    return model

# Crearea modelului
model = generate_model()

# Compilare model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

from tensorflow.keras.callbacks import EarlyStopping

# Creeaza callback-ul EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',       # Monitorizează pierderea pe setul de validare
    patience=8,               # Oprește dacă nu există îmbunătățiri timp de 3 epoci
    restore_best_weights=True # Restaurează greutățile cele mai bune
)

# Antrenare model cu EarlyStopping
history = model.fit(
    train_dataset,            # Setul de antrenament
    epochs=30,                # Număr maxim de epoci
    validation_data=val_dataset, # Setul de validare
    callbacks=[early_stopping]   # Callback EarlyStopping
)

# Evaluare model pe setul de testare
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()