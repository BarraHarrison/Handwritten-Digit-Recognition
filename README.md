# Handwritten-Digit Recognition using Python (Tensorflow)
This Python project recognizes handwritten digits (numbers).
The MNIST dataset was used to train the model and it can predict digits based on images (pngs) which are provided in the digits directory.

# Overall Description of the Project
- Load and preprocess the MNIST dataset
- Build and train a deep learning model using tensorflow/keras
- Save the model (Handwritten.keras)
- OpenCV and Numpy Libraries used to predict the digits
- Matplotlib used to visualize the results

# How the code works

## Libraries Imported
- OpenCV as cv2: Loading and preprocessing custom made digit images
- Numpy: Array manipulation (img = np.invert(np.array([img])))
- Matplotlib: Visualization (plt.imshow(img[0], cmap=plt.cm.binary))
- Tensorflow: Building and training the model
- os: Checking if the file exists

## Loading and Preprocessing the MNIST dataset
- MNIST dataset contains 60,000 training images and 10,000 testing images

## Defining and Training the Model
- This code can be commented out once the model is trained and saved
- Flatten converts 2D images into 1D arrays
- Dense layers for relu activation
- One Dense layer for rounding up (softmax)
- Model compiled with "adam" optimizer
- The model is trained for 3 epochs (epochs are just iterations)

## Loading the Saved Model
- The saved model is reloaded (Handwritten.keras)
- No need to retrain the model

## Predicting the Handwritten Digits
- Python script processes the digit images stored in the digits directory
- Image loaded using OpenCV (cv2.imread)
- Numpy used to invert the images
- Image converted into a Numpy array for prediction
- The model predicts the digit in the image as the image is displayed
- This continues until all digits in the digit directory are processed