# Masked Face Detection

This project implements a face mask detection system using LBP and HOG features with a Support Vector Machine (SVM) classifier.

## Features
- Face detection using OpenCV's Haar Cascades.
- Feature extraction using Local Binary Patterns (LBP) and Histogram of Oriented Gradients (HOG).
- Classification using a trained SVM model.
- Real-time detection via webcam.

## Files
- `train.py`: Script to train the SVM model on the dataset.
- `detect.py`: Real-time face mask detection using the trained model.
- `mask_classifier.pkl`: Pre-trained SVM model weights.
- `generate_docx_report.py`: Script to generate a project report in DOCX format.

## Setup
1. Install dependencies:
   ```bash
   pip install opencv-python scikit-learn scikit-image numpy
   ```
2. Run training (optional if `mask_classifier.pkl` exists):
   ```bash
   python train.py
   ```
3. Run detection:
   ```bash
   python detect.py
   ```
