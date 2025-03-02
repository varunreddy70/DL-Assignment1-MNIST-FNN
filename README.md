# Implementation of a Feedforward Neural Network (FNN) for MNIST Digit Classification

## Overview
This project implements a Feedforward Neural Network (FNN) for digit classification using the MNIST dataset. The model is built using TensorFlow/Keras and optimized with various hyperparameters.

## Dataset
- **Name:** MNIST
- **Images:** 60,000 training, 10,000 test
- **Size:** 28x28 pixels (grayscale)
- **Classes:** 10 (digits 0-9)

## Model Architecture
- **Input:** 784 neurons (flattened 28x28 image)
- **Hidden Layers:** 3 layers, 128 neurons each (ReLU activation)
- **Output Layer:** 10 neurons (Softmax activation)
- **Optimizer:** Adam, SGD, RMSprop
- **Loss Function:** Cross-Entropy Loss, Squared Error Loss

## Installation
Ensure you have Python 3.x installed, then install the dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Usage

Clone the repository and navigate to the project folder:

```bash
git clone https://github.com/varunreddy70/DL-Assignment1-MNIST-FNN/
cd DL-Assignment1-MNIST-FNN
```

## Training the Model

### Open the Jupyter Notebook:

```bash
jupyter notebook MNIST_FNN_Classification.ipynb
```
Run the notebook cells sequentially to:

- Load and preprocess the MNIST dataset.  
- Build the Feedforward Neural Network (FNN) model.  
- Train the model using the specified hyperparameters (e.g., learning rate, batch size, optimizer).  
- Monitor training progress, including loss and accuracy metrics.  

## Evaluating the Model

After training, the notebook will automatically evaluate the model on the test set.

The evaluation results will include:

- **Test accuracy.**  
- **Loss values** (cross-entropy or squared error).  
- **A confusion matrix** to visualize model performance.  

## Hyperparameter Tuning

To experiment with different hyperparameters (e.g., learning rate, batch size, optimizer), modify the relevant sections in the notebook and rerun the training and evaluation steps.


## Results
- **Best Model Accuracy:** 97.67%
- **Loss Comparison:**
  - Cross-Entropy Loss: 0.0934
  - Squared Error Loss: 0.0046

## References
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
