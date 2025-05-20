# PCA for Noise Reduction in Handwritten Digits

This project demonstrates how to use **Principal Component Analysis (PCA)** to remove noise from images of handwritten digits. The goal is to reconstruct cleaner versions of noisy digit images using PCA's dimensionality reduction capabilities.

## 📌 Description

The dataset used is the `digits` dataset from `scikit-learn`, which contains 8x8 grayscale images of handwritten digits (0–9). We:
1. Add artificial Gaussian noise to the images.
2. Apply PCA to reduce dimensionality and capture the most important patterns.
3. Reconstruct the images from the reduced space.
4. Compare original, noisy, and reconstructed images visually and quantitatively using **Mean Squared Error (MSE)**.


## 🔧 Requirements

Make sure you have the following installed:

- Python 3.x
- Libraries listed in `requirements.txt`


## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/mohammadseyfi11/ML_Algorithms.git
   cd ML_Algorithms
