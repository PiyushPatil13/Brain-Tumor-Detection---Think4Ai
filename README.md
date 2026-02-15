

# Brain Tumor Detection - Think4AI

This project implements a deep learning-based system for detecting brain tumors from MRI images using Convolutional Neural Networks (CNNs). The model is trained on a dataset of brain MRI scans to classify images as containing a tumor or not. It leverages TensorFlow/Keras for model development and includes scripts for data preprocessing, training, evaluation, and prediction. An optional web interface allows users to upload images for real-time detection.

### Features

1. Data Preprocessing: Automated loading, resizing, normalization, and augmentation of MRI images to prepare them for training.

2. Model Training: Custom CNN architecture with convolutional layers, pooling, dropout for regularization, and fully connected layers for classification.

3. Evaluation Metrics: Comprehensive evaluation including accuracy, precision, recall, F1-score, confusion matrix, and ROC curves.

4. Prediction Tools: Scripts for making predictions on individual images or batches.

5. Visualization: Plots for training loss/accuracy curves, sample predictions, and model performance.

6. Web Application: Javascript web app for uploading MRI images and viewing detection results.

### Installation

1. Clone the repository

```bash
git clone https://github.com/PiyushPatil13/Brain-Tumor-Detection---Think4Ai.git
cd Brain-Tumor-Detection---Think4Ai
```
2. Download the necessary libraries

```bash
pip install torch torchvision numpy pillow opencv-python pytorch-grad-cam optuna matplotlib
```

### Usage

Insert an image on the website , wait for the image to process , recieve confidence , gradcam image and type of tumors

### Dataset

1. Source: Brain MRI Images for Brain Tumor Detection (or similar public datasets).

2. Structure: Dataset should be split into training/ and testing/ directories, each containing subdirectories for classes (e.g., tumor and no_tumor).

3. Preprocessing: Images are resized to 224x224 pixels, normalized to [0,1], and augmented (e.g., rotations, flips) during training to improve generalization.

### Contributing

Contributions are welcome! Fork the repository, create a feature branch, and submit a pull request. Please ensure code adheres to PEP8 standards, includes docstrings, and passes any existing tests. For major changes, open an issue first to discuss.

### Acknowledgements

1. Dataset sourced from Kaggle and other public repositories.

2. Inspired by research in medical image analysis and deep learning for healthcare, including papers on CNN-based tumor detection.

3. Thanks to the open-source community for libraries like TensorFlow and Flask.
