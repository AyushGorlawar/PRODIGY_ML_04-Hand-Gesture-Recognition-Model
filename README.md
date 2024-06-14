# PRODIGY_ML_04
# Hand Gesture Recognition Model

This project develops a hand gesture recognition model using a dataset of hand gestures. The model can be used for intuitive human-computer interaction and gesture-based control systems.

# Gesture Recognition using Convolutional Neural Networks

This project aims to recognize hand gestures using a Convolutional Neural Network (CNN). The dataset used is the LeapGestRecog dataset, which contains images of different hand gestures.

## Project Structure

.
├── data
│ └── leapGestRecog
│ ├── 00
│ │ ├── frame_00_0000.png
│ │ ├── ...
│ ├── 01
│ │ ├── frame_01_0000.png
│ │ ├── ...
│ └── ...
├── gesture_model.h5
├── main.py
└── README.md

- `data/leapGestRecog`: Directory containing the dataset.
- `gesture_model.h5`: The trained model saved in HDF5 format.
- `main.py`: Main script to load data, train the model, and evaluate performance.
- `README.md`: This readme file.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Keras
- TensorFlow
- scikit-learn

Install the required packages using the following command:

```bash
pip install opencv-python-headless numpy matplotlib keras tensorflow scikit-learn
```
Download the dataset from [Kaggle](https://www.kaggle.com/gti-upm/leapgestrecog) and unzip it into the `data/` directory.

## Outputs
