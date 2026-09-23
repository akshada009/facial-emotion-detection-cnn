# 🤖 Live Facial Emotion Detection Using CNN

A machine learning project that detects human facial emotions in real time using a webcam. The system uses a Convolutional Neural Network (CNN) trained on facial expression images and OpenCV for live face detection.

## 📌 Project Overview

This project classifies facial expressions into seven emotion categories:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The trained CNN model processes facial images and predicts the most likely emotion. OpenCV is used to detect faces through the webcam for real-time emotion prediction.

## ✨ Features

- Facial expression classification using CNN
- Seven emotion categories
- Image preprocessing and normalization
- Model training and validation
- Test dataset evaluation
- Accuracy and loss visualization
- Confusion matrix for performance analysis
- Real-time webcam emotion detection

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook

## 🧠 Machine Learning Model

A Convolutional Neural Network (CNN) is used because CNNs are well suited for image classification tasks.

The model contains:

- Convolutional layers
- Max Pooling layers
- Dropout layers
- Fully connected Dense layers
- Softmax output layer

## 📂 Project Structure

```text
facial-emotion-detection-cnn/
│
├── facial_emotion_detection.ipynb
├── train_model.py
├── test_model.py
├── live_emotion.py
├── requirements.txt
└── README.md
```

## 📁 Dataset

The project uses a FER2013-based facial expression dataset containing seven emotion classes.

The dataset is not included in this repository because of its size.

## 🖥️ Live Application Preview

![Live Facial Emotion Detection](live_emotion_detection.png)


## 📊 Model Performance

The initial CNN model achieved approximately **54% test accuracy** on the test dataset.

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

### 3. Test the model

```bash
python test_model.py
```

### 4. Run live emotion detection

```bash
python live_emotion.py
```

The webcam will open and detect faces with their predicted emotions.

## 📈 Evaluation

The project includes:

- Training and validation accuracy
- Training and validation loss
- Test accuracy
- Confusion matrix
- Actual vs predicted emotion comparison

## 🎯 Learning Outcomes

- Image preprocessing and normalization
- CNN-based image classification
- Model training and evaluation
- Understanding overfitting
- Confusion matrix analysis
- Real-time face detection using OpenCV
- Live facial emotion prediction
