import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Dataset
train_data = ImageDataGenerator(rescale=1./255)
val_data = ImageDataGenerator(rescale=1./255)

train = train_data.flow_from_directory(
    "data/train",
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode="categorical"
)

val = val_data.flow_from_directory(
    "data/val",
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode="categorical"
)

# Simple CNN
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(7, activation="softmax")
])

# Train
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train, validation_data=val, epochs=10)

# Save model
model.save("emotion_model.keras")

print("Model saved successfully!")