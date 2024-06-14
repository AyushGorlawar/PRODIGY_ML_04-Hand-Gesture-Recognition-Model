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
![Model Accuracy](https://www.kaggleusercontent.com/kf/168300517/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mjAcVx0sDSpLo9DGzGe17w.29sWtQbVGW_CKKTni7j4kuX8ubXoHWg-xOD3sgBXrbwpPfxjdsrP5wECVA_HqJuZyLjaASMQjz5lLY85qVF2Xq7Yz12uoZXFzexFiCu3iEMMtxtBVWs8lCM23zqSPh9UAs9-mETwWOw3R9TktYTzmrVsBL_1cF84vuwF1bB1jJKrye4gUlRlNIazdXxhlmbZJhfhpLKg-1JwP5sNTJvENXvURA0IOk416W-4VFu5py0GzEAJp_UOc6-iiwiruSWKILh9oZ8iYTOGvL-UfmrspHTAIwlFlikDTmw8Z_JiuarG150dBLG3diCi3xIhAJse3DefkvyJOtvqT7J29TC8Z-U7jppvomHDi4NmBi7hkdn7KrnhWR_DCMlPb-iVN8zxI9Gqq2lEst73fvA2DBdL7sjM2-9tLBEGXKrGg843jpfqMJkglqRMgvlqZLI53ojAWBypPu9tS2GdxbpwyL-yy2csFhpv_T0q4hnIWm5cXY5_tXAoklDNf1HLDspyO-Uc8rO3dGO_cy5bb0bPrL6mGwn-K-m_8Qp6P4xzkMUTumqklPeNkAIpmhtsRrK0C6tZrbxRRADH18u2akQnjKx2Kyg_kAusF5oRkxgy-8hGiZohZT8y3NeaU3hMZ5CDXOyosTp8UhKS6jqeVwKXufTOKHZaXdsTw5hpAuSLdG9ou4M.VwR8-E6VH0Qe8lDAnbvv-Q/__results___files/__results___34_0.png)

![Model Loss](https://www.kaggleusercontent.com/kf/168300517/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mjAcVx0sDSpLo9DGzGe17w.29sWtQbVGW_CKKTni7j4kuX8ubXoHWg-xOD3sgBXrbwpPfxjdsrP5wECVA_HqJuZyLjaASMQjz5lLY85qVF2Xq7Yz12uoZXFzexFiCu3iEMMtxtBVWs8lCM23zqSPh9UAs9-mETwWOw3R9TktYTzmrVsBL_1cF84vuwF1bB1jJKrye4gUlRlNIazdXxhlmbZJhfhpLKg-1JwP5sNTJvENXvURA0IOk416W-4VFu5py0GzEAJp_UOc6-iiwiruSWKILh9oZ8iYTOGvL-UfmrspHTAIwlFlikDTmw8Z_JiuarG150dBLG3diCi3xIhAJse3DefkvyJOtvqT7J29TC8Z-U7jppvomHDi4NmBi7hkdn7KrnhWR_DCMlPb-iVN8zxI9Gqq2lEst73fvA2DBdL7sjM2-9tLBEGXKrGg843jpfqMJkglqRMgvlqZLI53ojAWBypPu9tS2GdxbpwyL-yy2csFhpv_T0q4hnIWm5cXY5_tXAoklDNf1HLDspyO-Uc8rO3dGO_cy5bb0bPrL6mGwn-K-m_8Qp6P4xzkMUTumqklPeNkAIpmhtsRrK0C6tZrbxRRADH18u2akQnjKx2Kyg_kAusF5oRkxgy-8hGiZohZT8y3NeaU3hMZ5CDXOyosTp8UhKS6jqeVwKXufTOKHZaXdsTw5hpAuSLdG9ou4M.VwR8-E6VH0Qe8lDAnbvv-Q/__results___files/__results___35_0.png)

![Confusion Matrix](https://www.kaggleusercontent.com/kf/168300517/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mjAcVx0sDSpLo9DGzGe17w.29sWtQbVGW_CKKTni7j4kuX8ubXoHWg-xOD3sgBXrbwpPfxjdsrP5wECVA_HqJuZyLjaASMQjz5lLY85qVF2Xq7Yz12uoZXFzexFiCu3iEMMtxtBVWs8lCM23zqSPh9UAs9-mETwWOw3R9TktYTzmrVsBL_1cF84vuwF1bB1jJKrye4gUlRlNIazdXxhlmbZJhfhpLKg-1JwP5sNTJvENXvURA0IOk416W-4VFu5py0GzEAJp_UOc6-iiwiruSWKILh9oZ8iYTOGvL-UfmrspHTAIwlFlikDTmw8Z_JiuarG150dBLG3diCi3xIhAJse3DefkvyJOtvqT7J29TC8Z-U7jppvomHDi4NmBi7hkdn7KrnhWR_DCMlPb-iVN8zxI9Gqq2lEst73fvA2DBdL7sjM2-9tLBEGXKrGg843jpfqMJkglqRMgvlqZLI53ojAWBypPu9tS2GdxbpwyL-yy2csFhpv_T0q4hnIWm5cXY5_tXAoklDNf1HLDspyO-Uc8rO3dGO_cy5bb0bPrL6mGwn-K-m_8Qp6P4xzkMUTumqklPeNkAIpmhtsRrK0C6tZrbxRRADH18u2akQnjKx2Kyg_kAusF5oRkxgy-8hGiZohZT8y3NeaU3hMZ5CDXOyosTp8UhKS6jqeVwKXufTOKHZaXdsTw5hpAuSLdG9ou4M.VwR8-E6VH0Qe8lDAnbvv-Q/__results___files/__results___37_0.png)

# - License
 
## This project is licensed under the [MIT License.](<https://github.com/AyushGorlawar/PRODIGY_ML_04/blob/main/LICENSE>)
