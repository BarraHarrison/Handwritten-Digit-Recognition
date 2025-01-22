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
