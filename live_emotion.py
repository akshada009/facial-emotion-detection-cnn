import cv2
import numpy as np
import tensorflow as tf

# =========================
# Load trained CNN model
# =========================

model = tf.keras.models.load_model(
    r"D:\Facial_Emotion_Detection\emotion_model.keras"
)

# Emotion labels
emotions = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# =========================
# Load face detector
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Face detector could not be loaded.")

# =========================
# Open webcam
# =========================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================
# Live emotion detection
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve brightness/contrast
    gray = cv2.equalizeHist(gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Process every detected face
    for (x, y, w, h) in faces:

        # Crop face
        face = gray[y:y+h, x:x+w]

        # Resize to model input
        face = cv2.resize(face, (48, 48))

        # Normalize
        face = face.astype(np.float32) / 255.0

        # Shape -> (1, 48, 48, 1)
        face = face.reshape(1, 48, 48, 1)

        # CNN prediction
        prediction = model.predict(
            face,
            verbose=0
        )

        emotion = emotions[np.argmax(prediction)]

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Background for emotion text
        cv2.rectangle(
            frame,
            (x, y - 40),
            (x + w, y),
            (0, 255, 0),
            -1
        )

        # Emotion text
        cv2.putText(
            frame,
            emotion,
            (x + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )

    # Show live video
    cv2.imshow(
        "Live Facial Emotion Detection",
        frame
    )

    # Press Q to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# =========================
# Release resources
# =========================

cap.release()
cv2.destroyAllWindows()