Implementation of a Feedforward Neural Network (FNN) for MNIST digit classification.

## Overview
This project implements a Feedforward Neural Network (FNN) for digit classification using the MNIST dataset. The model is built using TensorFlow/Keras and optimized with various hyperparameters.

## Dataset
- Name: MNIST
- Images: 60,000 training, 10,000 test
- Size: 28x28 pixels (grayscale)
- Classes: 10 (digits 0-9)

## Model Architecture
- Input: 784 neurons (flattened 28x28 image)
- Hidden Layers: 3 layers, 128 neurons each (ReLU activation)
- Output Layer: 10 neurons (Softmax activation)
- Optimizer: Adam, SGD, RMSprop
- Loss Function: Cross-Entropy Loss, Squared Error Loss

## Installation
Ensure you have Python 3.x installed, then install the dependencies:
pip install tensorflow numpy matplotlib scikit-learn

## Usage
Clone the repository and navigate to the project folder:
git clone <repo-link>
cd deep-learning-mnist

Run the Jupyter Notebook:
jupyter notebook DL_Assignment_1.ipynb

## Results
- **Best Model Accuracy:** 97.67%
- **Loss Comparison:**
  - Cross-Entropy Loss: 0.0934
  - Squared Error Loss: 0.0046

## References
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
