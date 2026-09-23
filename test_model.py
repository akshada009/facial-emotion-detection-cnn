import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load trained model
model = tf.keras.models.load_model("emotion_model.keras")

# Prepare test data
test_data = ImageDataGenerator(rescale=1./255)

test_generator = test_data.flow_from_directory(
    "data/test",
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=False
)

# Evaluate model
loss, accuracy = model.evaluate(test_generator)

print("Test Accuracy:", accuracy)
print("Test Loss:", loss)