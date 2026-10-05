# Blood Group Detection Using Fingerprint

This project detects blood groups using fingerprint images with a ResNet-based deep learning model. The application provides a web interface where users can upload fingerprint images and get blood group predictions.

## Features

- ResNet-based deep learning model for accurate predictions
- Web interface for easy interaction
- Enhanced fingerprint image preprocessing using CLAHE
- Top 3 predictions with confidence scores
- Responsive design that works on desktop and mobile devices
- Calibrated confidence scores for more reliable predictions

## Requirements

- Python 3.7 or higher
- Flask
- TensorFlow
- OpenCV
- NumPy

## Installation

1. Clone or download this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Downloading the Pre-trained Model

Due to GitHub's file size limitations, the pre-trained model file (`calibrated_model.h5`) is not included in this repository. You can download it from the releases section:

1. Visit the [Releases page](https://github.com/chaitu0-12/Blood_group_detection/releases)
2. Download the `calibrated_model.h5` file
3. Place the downloaded file in the project root directory

Alternatively, you can train your own model using the provided training scripts.

## Usage

1. Ensure you have the `calibrated_model.h5` file in the project root directory
2. Run the application:
   ```
   python app.py
   ```
3. Open your web browser and go to `http://localhost:5000`
4. Upload a fingerprint image (PNG, JPG, or JPEG format)
5. View the predicted blood group and confidence score

## Blood Groups Supported

The model can predict the following 8 blood groups:
- A+
- A-
- B+
- B-
- AB+
- AB-
- O+
- O-

## Model Information

The application uses a ResNet-based convolutional neural network that has been trained on fingerprint images to classify blood groups. The model expects input images of size 224x224 pixels.

## Preprocessing Techniques

To improve accuracy, the application applies the following preprocessing techniques:
- Color space conversion (BGR to RGB)
- Image resizing to 224x224 pixels
- Contrast Limited Adaptive Histogram Equalization (CLAHE) for fingerprint enhancement
- Pixel normalization to [0, 1] range

## Disclaimer

This prediction is based on machine learning analysis. For medical purposes, please confirm with a blood test.