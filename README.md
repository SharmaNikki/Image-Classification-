# Project Description : Image-Classification
1. Naïve Bayes
2. Random Forest
3. Deep Learning
4. Support Vector Machines
	
The models takes in images and their labels, then classify each image


# Set GPU Environment
  - In order to get fast computing, we can't use CPU processor to process our data because the size is too large. 
    Therefore the models were run on the MS-Azure Virtual Machine hosting R to compute the performance of model processing.
  - Azure virtual machines (VMs) can be created through the Azure portal. This method provides a browser-based user interface to create     VMs and their associated resources. 
  - The VM resouces used was 56 GB RAM - with capacity of 200 GB HD

# Data Set

- In this project we have build different  model for image classification to distinct places such as buildings, forest, glacier,      mountain, sea, and street with around 25k images for train data
- For each scenes, it has individual set of images, for to train and the test model. Most of the images are in 150 x 150 pixle size and   some where in improper size format. So, all the images will be resized to 128 x 128.
-This model can be used for application that used landscape picture as its own features for instance to cluster recommendation places that similar with user input.

# Importing Libraries
We need to import several packages but mostly we just need packages for data manipulation and build deep learning architecture model.
- Packages : install.packages("BiocManager") for loading the EBImage library

###  Libraries required 
- library(keras)
- library(tensorflow)
- library(EBImage)

# Data Analysis 

Firstly, before we fit our data into CNN model, we have to serve it as a matrix form. The images are converted into multi dimensional  Matrix (width, height, and channel). In the dataset folder, there are 3 subfolders contain train, test, and prediction dataset. In this case we're only using  2 subfolders, seg_train for making train set and validation set, and seg_test to evaluate our model.

## Technique

## Neural Network 
### Labeled the data
  Feedforward DNNs require all feature inputs to be numeric. Consequently, if the data contains categorical features they will need to     be numerically encoded (e.g., one-hot encoded, integer label encoded, etc.)
### One hot Encoding 
  Data to be numerically encoded
  
## Neural Network - 
### Build Model in Keras-Tensorflow
- We initiated our sequential feedforward DNN architecture with keras_model_sequential() and then add some dense layers
- The first with 256 nodes and the second with 128, followed by an output layer with 6 nodes.
- First layer needs the input_shape argument to equal the number of features in the data; however, the successive layers are able to       dynamically interpret the number of expected inputs based on the previous layer.
- When using rectangular data, the most common approach is to use ReLU activation functions in the hidden layers. The ReLU activation     function is simply taking the summed weighted  inputs.
### Compile 
 - To incorporate the backpropagation piece of our DNN we include compile()
 
 <a href="https://imgbb.com/"><img src="https://i.ibb.co/jHXXZvT/image.png" alt="image" border="0"></a>
 
### Fit the Model 
 - We’ve created a base model, now we just need to train it with some data. To do so we feed our model into a fit() function along with our training data

## Support Vector Machine 
 Support Vector Machine test in R was used to perform the image classification test.The below steps were performed to evaluate the model performance 
###  Below are the steps followed for to train & test the model.
- Import the necesasary libraties and image files
- Create a Linear SVM model and train it
- Check the accuracy score of the model
### Training data importing with labelling 
### Fitting SVM 
### Predict SVM 
### ROC Curve
- To derive the ROC curve from the probability distribution, we calculated the True Positive Rate (TPR) and False Positive Rate (FPR).
<a href="https://imgbb.com/"><img src="https://i.ibb.co/ZWQ4Xtb/image.png" alt="image" border="0"></a>

## Naive Bayes 
 Naive Bayes test in R was used to perform the image classification test.The below steps were performed to evaluate the model performance 
### Training data importing with labelling 
### Fitting Naive Bayes 
### Predict Naive Bayes 
### ROC Curve
- To derive the ROC curve from the probability distribution, we calculated the True Positive Rate (TPR) and False Positive Rate (FPR).
<a href="https://imgbb.com/"><img src="https://i.ibb.co/0J0kD5q/image.png" alt="image" border="0"></a>

## Random Forest 
 Random Forest  test in R was used to perform the image classification test.The below steps were performed to evaluate the model performance 
### Training data importing with labelling 
### Fitting Random Forest  
### Predict Random Forest 
### ROC Curve 
- To derive the ROC curve from the probability distribution, we calculated the True Positive Rate (TPR) and False Positive Rate (FPR).
<a href="https://imgbb.com/"><img src="https://i.ibb.co/DfvWWb4/image.png" alt="image" border="0"></a>

## Hypothesis Testing 
 Anova test is utilised to compare the models performance and check the siginificant difference between the models
 ### ANOVA is used to determine whether there are any statistically significant differences between the means of three or more independent (unrelated) groups.
 <a href="https://imgbb.com/"><img src="https://i.ibb.co/XJfN7kZ/image.png" alt="image" border="0"></a>
 
 
 

 

