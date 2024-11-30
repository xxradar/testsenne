# testsenne

## Cat and Dog Classifier

This repository contains a simple PyTorch example to separate cats and dogs using a Convolutional Neural Network (CNN).

### Requirements

- Python 3.x
- PyTorch
- torchvision

### Instructions

1. Clone the repository:
   ```
   git clone https://github.com/xxradar/testsenne.git
   cd testsenne
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the cat_dog_classifier.py script:
   ```
   python cat_dog_classifier.py
   ```

### Model Explanation

The model used in this example is a simple Convolutional Neural Network (CNN) with the following architecture:

- Convolutional layer with 32 filters, kernel size 3x3, and ReLU activation
- MaxPooling layer with pool size 2x2
- Convolutional layer with 64 filters, kernel size 3x3, and ReLU activation
- MaxPooling layer with pool size 2x2
- Fully connected layer with 128 units and ReLU activation
- Output layer with 2 units (for cat and dog classification) and softmax activation

The model is trained on a dataset of cat and dog images, and the performance is evaluated on a test set.
