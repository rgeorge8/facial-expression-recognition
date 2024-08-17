# facial-expression-recognition
This project focuses on building a Facial Expression Recognition system using a deep learning approach. The model is trained to identify and classify facial expressions from images into one of seven categories (surprise, happy, neutral, sad, fear, disgust and angry)

## Dataset
The project utilizes the Facial Expression Dataset from Kaggle, which contains grayscale images of faces with annotated expressions. The images are 48x48 pixels in size, which are suitable for training deep learning models efficiently.

## Libraries and Tools
The project is implemented in a Kaggle notebook and leverages the following Python libraries:

- Pandas: For data manipulation and analysis.
- NumPy: For numerical computations and array operations.
- Matplotlib: For data visualization and plotting.
- Keras & TensorFlow: For building and training the deep learning model.
- Scikit-learn: For data preprocessing and model evaluation.
  
## Model Overview
The model is a Convolutional Neural Network (CNN) designed to process the 48x48 grayscale images and classify them into one of several facial expression categories. The network is trained using the categorical_crossentropy loss function and optimized with the adam optimizer. The accuracy of the model is tracked as the primary performance metric.
This project detects facial expressions using this image data set: https://www.kaggle.com/datasets/aadityasinghal/facial-expression-dataset
- Used Convolutional Neural Network (CNN) to classify the images. 
- The output class consists of 7 different types: 
