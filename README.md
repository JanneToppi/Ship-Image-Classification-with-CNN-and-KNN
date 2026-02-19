# Ship Detection from Satellite Imagery Using CNN and KNN

## Overview

This project was my first machine learning / deep learning project. 
The objective was to develop a supervised machine learning model capable of detecting ships in satellite images.

Two different classification approaches were implemented and compared:

- Convolutional Neural Network (CNN)
- k-Nearest Neighbors (KNN) with PCA-based dimensionality reduction

The task is a binary classification problem:
- 1 → Ship
- 0 → No Ship

---

## Dataset

- Source: Kaggle – Ships in Satellite Imagery (Hammell, 2018)
- 4,000 labeled training images
  - 1,000 ship images
  - 3,000 no-ship images
- Image size: 80x80 RGB
- Satellite locations: San Francisco Bay and San Pedro Bay

Images were provided as `.png` files with associated metadata.

---

## Methodology

## Data Preprocessing

- Pixel normalization to range [0,1]
- Label encoding (Ship = 1, No Ship = 0)
- Data augmentation for CNN:
  - Rotation
  - Horizontal flipping
  - Zoom
  - Width/height shifts
- Dataset split:
  - 80% Training
  - 10% Validation
  - 10% Test

---

## Model 1: Convolutional Neural Network (CNN)

Architecture:

- 3 Convolutional layers (ReLU activation)
- Max pooling layers
- Fully connected dense layer (128 units)
- Dropout (0.5) for regularization
- Softmax output layer (2 classes)

Loss function:
- Categorical Cross-Entropy

Optimizer:
- Adam

Training:
- 11 epochs
- Early stopping tested for overfitting control

### CNN Test Performance

- Test Accuracy: 97.00%
- Test Loss: 0.0971
- Test Error: 3.00%

Classification performance:

- No Ship:
  - Precision: 0.99
  - Recall: 0.97
- Ship:
  - Precision: 0.91
  - Recall: 0.98

The confusion matrix shows strong performance with very few false negatives and false positives.

---

## Model 2: K-Nearest Neighbors (KNN)

Because raw pixel values create a very high-dimensional feature space, PCA was applied before KNN.

### Preprocessing for KNN

- Images flattened
- Standard scaling applied
- PCA with 50 components
  - ~84% explained variance

### Hyperparameter Tuning

GridSearchCV used for:
- n_neighbors (1–30)
- Distance metric (Euclidean, Manhattan)
- Weighting (uniform, distance)

Best parameters:
- n_neighbors = 2
- metric = Manhattan
- weights = uniform

### KNN Performance

- Accuracy: ~97%
- Precision (Ship): 0.92
- Recall (Ship): 0.95
- F1 Score: 0.94

KNN performed better than expected and achieved comparable accuracy to CNN.

---

## Model Comparison

Both models achieved strong results:

- CNN slightly outperformed KNN overall.
- KNN achieved slightly higher precision in some cases.
- CNN demonstrated better generalization and scalability for complex image tasks.

Final model selected: CNN

## Conclusion

Both CNN and KNN are viable approaches for ship detection from satellite imagery.

CNN proved to be more suitable for image-based tasks due to its ability to learn hierarchical spatial features directly from raw data.

While KNN performed surprisingly well after dimensionality reduction, high-dimensional image data inherently favors deep learning approaches.

Further improvements could include:
- More advanced CNN architectures
- Regularization tuning
- Better early stopping criteria

