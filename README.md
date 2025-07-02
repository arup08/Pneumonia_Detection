# 🧪 Pneumonia Detection Using Deep Learning (PyTorch)

A deep learning-based binary classification system for detecting **Pneumonia** from chest X-ray images using **Convolutional Neural Networks (CNNs)**.

## 📌 Overview

This project uses a custom CNN trained on grayscale chest X-ray images to classify cases as either:

- **NORMAL** – Healthy lungs
- **PNEUMONIA** – Infected lungs

Early detection through AI helps assist radiologists, especially in low-resource healthcare settings.

## 🫁 About Pneumonia

Pneumonia inflames the lungs' air sacs and can fill them with fluid. Early symptoms include:

- Cough and fever  
- Chest pain and shortness of breath  
- Fatigue and chills  

**Early AI-based detection** can reduce mortality by improving diagnosis speed and consistency.

## 🧠 Model Architecture

- **Input Size**: 150×150 grayscale images  
- **Layers**: 5 convolutional blocks  
  - Conv (3×3) + ReLU + BatchNorm + MaxPool + Dropout  
- **Fully Connected Layers**:  
  - Flatten → Dense(128) → Sigmoid  
- **Total Parameters**: ~1.4 million

## ⚙️ Implementation Details

### 🖼️ Data Preprocessing

- Grayscale conversion
- Resized to 150×150
- Normalized (mean = 0.5, std = 0.5)
- Augmentation: Random transformations (flip, crop)
- Managed using `PyTorch DataLoader`

### 🏋️ Training

- **Loss Function**: Binary Cross-Entropy Loss  
- **Optimizer**: RMSprop  
- **Learning Rate**: 0.001  
- **Batch Size**: 32  
- **Epochs**: 10  

### 📈 Performance

- **Validation Accuracy**: **87.25%**
- **Metrics Tracked**: Accuracy, loss curves, confusion matrix
