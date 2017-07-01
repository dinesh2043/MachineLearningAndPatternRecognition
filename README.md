## Machine Learning and Pattern Recognition
### 1 Exercise 1: Pattern recognition of handwritten characters
### 1.1	Introduction
This exercise was done using MNIST handwritten digits dataset available on web, which consists 60,000 training set and 10,000 test set. This dataset is available as a compressed files of training set, test set and appropriate labels files. These datasets are normalized 28x28 images collected from 500 different writers which have been distributed into equal proportion of training set and test set. Due to these properties this database has been used in large varieties of machine learning application research. [1]
The available dataset are included into four different compressed files as training set, test set, training set label and test set label. Python gzip and struct library are used to extract data available in magic format and then it is converted to numerical format. Data obtained after this procedures are stored in four different variables called train_lbl, train_img, test_lbl and test image. In the following code snipped we can see the implementation of this section; [2]
 
#### Figure 1: Implementation to read the data from the compressed file
Particular image data consists a 28x28 matrix with values ranges from 0 to 255, where 0 is a black pixel and 255 is a white pixel. These available dataset has been plotted as training image and testing image to understand the visual representation of the values of this pixel in an image. We can apply the following implementation to 20 different training and testing images along with their label; <br />

   
#### Figure 2: Visual representation of test and train image
Each image of both test and training dataset are in the form of 28x28 matrix with values ranging from 0 to 255. If we plot an image in this format, it will produce a graph with 28 different lines each representing the corresponding values in its rows. In the following graph we can see the result when a matrix is plotted; <br />
 
#### Figure 3: Graph of an image in matrix format.
Image vectorization technique is used to reduce the multidimensional raw data into one dimensional data which might be useful in understanding the inner structure and key features of the data. In OpenCV library we can use flatten() method to obtain this goal. Since, we have our training set as an array of training image data, I have defined a vector() method in my implementation to provide a complete array of an image matrix and then convert that array of image matrix into vector of image. In the following section of this code we can see the process how it is done; <br />
 
#### Figure 4: Vectorization of image matrix into image vector.    
After vectorization we will obtain a one dimensional image with 784 different values and when we plot this we will get a graph where we can find the key information about the distribution of the values in this particular image. Since, our task is to develop a model which classifies handwritten digits and this vectorised image graph will help us to determine features for classification. In the following image we can see the graph obtained after vectorization; <br />
 
#### Figure 5: Graph of vectorised image.
### 1.2	Cluster Visualization 
Digit image beautification will have a great role in the cluster visualization because it will helps us to reduce the variance in the data of same digits. Which will help us to visualize our image where the data points belonging to the same cluster will come closer to each other. There are large varieties of image beautifying algorithm available for image processing. Some of the example of those process can be changing colour spaces, geometric transforming of images, image thresholding, smoothing images, morphological transformations, image edge detection, image de-skewing etc. In this particular exercise I have used image thresholding, image reduction and image de-skewing. Among those two technique image de-skewing will have a great impact in the cluster visualization because it will help us to reduce the unnecessary tilts of the handwritten digits which will result the cluster of the same digits to be closely packed. In the following figure we can see the method written to de-skew the image; [3] <br />
 
#### Figure 6: De-skewing method for digit image
As a result of this method all the image will be aligned in the centre of the image and the digit are aligned straight in the centre point. It can be seen in the following image; <br />
 
 
#### Figure 7: Result in the digit image after de-skewing them. 
Skeleton line matching can also be used as a technique for dimensionality reduction of the image because it preserves the structure and orientation of the image. Which also result to have a better performance of the model, but it will not have a significant effect on cluster visualization like image de-skewing. [3] <br />
### 1.3	Feature Extraction
Dimension reduction was implemented to extract the key features of the digit images. As, we know that the key data required for image classification was present in the central part of 28x28 image. It was possible to reduce the pixel size of image to 14x14, without losing the key information of the image. Due to that reason both the training image and test image was reduced to half of the actual image size. In the following section of the code snippet we can see the method used for image reduction; <br />
 
#### Figure 8: Image reduction method implementation
After implementing this method we can plot the image obtained by its implementations, it is also a key step for efficient performance of the model. In the following picture we can see the implementation to plot the image and the final result in the image; <br />
 
#### Figure 9: Resulting image after dimensionality reduction
Image thresholding was used to have a better features for creating better feature extractor for the classification model. Where the pixels having the values less than 127 were set to be 0 as dark pixel and the pixel values greater than 127 were set to 255 as white pixel. Which also helps to increase the performance of the model. In the following picture we can see the implementation and the result of image thresholding in digits image; <br />
 
#### Figure 10: Result in the digit image after image thresholding
Template matching is the process of finding the location of the template image in a larger image. This process was performed by using 30 different original image where 3 images of each digits was used as large image and 3 images of each digits obtained after image reduction are used as template. In the following figure you can see the images used; <br />
     
     
     
     
#### Figure 11: Large image and template image used in the exercise.
OpenCV library has a method called matchTemplate() which calculates all the parameters which are required for the template matching algorithm. To accomplish this result I have implemented a method called templateMatching where the large image, template image and the index of the image is supplied to perform the matching. In the following code snippet we will be able to see the complete implementation; <br />
 
#### Figure 12: Method implementation for the template matching 
In the following pictures we will be able to see the results of template matching on the original image with the help of template in each 3 digits image; <br />
     
   
    
 
#### Figure 13: Template matching results of all 10 digits image
After the image reduction to 14x14 pixel, another feature extraction technique called Histogram of Oriented Gradients (HOG) was used to convert the grayscale image to a feature vector. To accomplish this task HOG descriptor parameters values called winSize, cellSize, blockSize, blockStride, nbins and signedGradients were defined to prepare the training and testing data for the model to accomplish better performance. In the following code snippet we will be able to see its implementation; [4] <br />
   
#### Figure 14: Implementation of HOG descriptor
Using this descriptor both the training and testing data was computed which was essential for creating the model for classification of digits image. In the following code snippet we will be able to see its implementation; <br />
  
#### Figure 15: Process of computing training and testing data
### 1.4	Classification
To solve the classification problem Support Vector Machines (SVM) algorithm was used as a supervised learning model. SVM is used as a training algorithm to build the model which classifies the test samples to one or other category according to the generalization of data based on the training sample. SVM model represents the training samples as a points in space such that the examples of separate categories are divided by a clear gap as wide as possible. When the test samples are supplied to the model, they are mapped into the same space and prediction is done on the basis of its position on the either side of the gap they belongs. [4]
Then the feature vectors computed using HOG descriptor were converted to float32 format. OpenCV 3 was used to create a model where, SVM model was defined along with its type. Then the kernel of SVM was set to be Radial Basis Function (RBF) and hyper parameter’s like C was set to be 12.5 and gamma to be 0.5 and model was trained. In the following code snipped we can see the implementation of this process; <br />
 
  
#### Figure 16: Steps for training the SVM model.
### 1.5	Model Performance
After the model is trained, it is tested with the test dataset to evaluate its performance. Our dataset also consisted the test data of 10,000 samples and these test data is supplied to the model to have a prediction of supplied test data set. In the following section we can see the code snipped of this implementation; <br />
 
#### Figure 17: Method to calculate the prediction of the model.  
All the test responses were stored in the in the variable called testResponse and the accuracy of the model was calculated to be 95.52% using the following code in the implementation; <br />
 
#### Figure 18: Calculation of the prediction accuracy of the model.
 Confusion Matrix is also known as error matrix, which is drawn in a table layout that helps us to visualize the performance of the algorithm. It consists the instances of the predicted classes in the column and the actual classes in the row. Pandas was used to take the advantage for batter visualization of the complete process. In the following picture we will be able to see the implementation and result of confusion matrix of the model; <br />
 
#### Figure 19: Confusion matrix of the model.
To find out the false positive and false negative results of the model following steps were performed to record all the miss-classifications. Its implementation can be seen in the following code snippet; <br />
 
#### Figure 20: Storing misclassification values.
Once those misclassified images were obtained, then they were plotted to have a better understanding why the model fail to classify them properly. In the following code snippet we can see the process used to have the visual representation; <br />
 
#### Figure 21: Miss-classified images with the actual and predicted values.
After looking at the images we can clearly see that these digits were written in bad format and they are not that clear even for human to recognize them. Due to this observation we can consider that the model has a good performance in-spite of these errors. Because machine learning is about generalization and we can see that the model has generalized well enough with the accuracy of 95.52 %. It has been possible due to the result of image processing, dimensionality reduction and feature extraction techniques implemented. The performance can also be fine-tuned using cross validation techniques but since, it was out of the scope of this exercise. It can be done for the further improvement of the model performance. It was a great learning experience while preforming these techniques to build the model using computer vision technique to solve the classification problem.


### 2	Exercise 2:
### 2.1	Deep Learning and Neural Network
Deep learning is the process of implementing machine learning with the help of artificial neural networks (ANNs) that consists more than one hidden layer. Perceptron and sigmoid neurons are the two key concepts of neural networks. Perceptron and developed in such a way that it takes several binary inputs and produces a single binary output. Perceptron can be defined as a mathematical model that makes decision based upon the evidence of their weights and threshold. Sigmoid neurons are similar to perceptron, but it is modified in such a way that small change in their weights and bias will cause only a small difference in its output. [6] Keras has been used as deep learning library because it is easier for the beginners due to its modular approach. Normally deep learning consists of 3 layers where first hidden layer learns local edge patterns, second layer learns more complex representation and the third layer classify the image. A multi-layer neural network that consist the input data as image is called Convolution Neural Networks (CCNs). [5]
To setup the environment python 3, SciPy with NumPy, Matplotlib, Theano and Keras was installed in the computer. The project implementation was started importing numpy library and the seed was to generate pseudo random numbers. Its implementation can be seen in the following code snippet; [5] <br />
    
#### Figure 22: Numpy import and random generator.
Then the sequential model was also imported from keras which, is important for the implementation of feed-forward convolution neural network. Similarly, the core layers called Dense, Droopout, Activation and Flatten were also imported. In addition to them convolution layer called Convolution2D and MaxPooling2D was also imported. Finally we also import utils for the transformation of data. Which can be seen in the following code snippet; [5] <br />       
#### Figure 23: Sequential model, keras core layers, utils and convolution layer imports.
### 2.2	Dataset Preparation
As, Keras library already includes the MNIST handwritten dataset which consists 60,000 training set with its label and 10,000 samples of test set with its label we can import them in our implementation. Then we store that data into X_train, y_train, X_test and y_test variables where X_train and X_test are digits images and y_train and y_test are the labels. We can check the shape of the training and testing dataset using python commands. We can refer to the following code snippet to see those process; [5] <br />
  
#### Figure 24: Implementation for storing datasets in different variables.
While working with the computer vision exercise it is important to pre-process the dataset to have a better performance of the data. Since, we are working on classification problem 3 RGB channel are not necessary so we can explicitly declare the depth to be 1 to reshape the dataset as one dimensional input image.  Reshaping of the dataset is done as follows; [5] <br />
 
#### Figure 25: Reshaping of the dataset. 
As, the final pre-processing step of the dataset we convert the data type to be float32 and also normalize the data values between the range [0, 1] where, 0 is the dark pixel and one is white pixel. This implementation can be seen in the following code snippet; [5] <br />
 
#### Figure 26: Data type and range setting
We have the labels of the training and test dataset in 1 dimensional array but we need 10 different classes for each digits. Then we need to convert it to 10-dimensional class matrix which can be done as follows; [5] <br />
 
#### Figure 27: Conversion of 1 dimensional array into 10-dimensional class matrix.
### 2.3	Model Architecture and its Performance
We were supposed to prepare a classification model which consists of the three layers first of all we define a sequential model. Then we define the CNN input layer with 32 convolution filter, 3 rows in the convolution kernel and 3 columns in each convolution kernel. We also define the input shape with depth value 1, width 28 and height 28. This CNN input layer uses relu as activation function and we can conform its implementation by printing the result. In the following code snippet we can see its implementation; [5] <br />
     
#### Figure 28: Declaration of input layer
Convolution layer also consists Convolution2D with relu activation function, Dropout is set to 0.25 to prevent overfitting and MaxPooling2D is set to 2x2 pooling filter in previous layer for taking the maximum of 4 values in 2x2 filter. In the following code snippet we can see its implementation; [5] <br />
 
#### Figure 29: Convolution layer implementation.
After adding those two convolution layer now we add a fully connected layer and the output layer in our implementation. First of all the weights of the convolution layer should be flattened to pass it to the connected dense layer. In dense layer 128 is used as the output size of the layer and keras handles the connection between the layers. In output layer the output size is set to 10 because we have 10 classes of digits. Its implementation can be seen in the following code snippet; [5] <br />
 
#### Figure 30: Fully connected and output layer.
Along with the above implementation our model architecture is ready now. After compiling this model, it can be trained. For the compilation of the model we need to declare the loss function, optimizer and matrices. Model compilation step can be seen in the following code; [5] <br />
   
#### Figure 31: Model compilation process.
Finally the model is ready to train where we provide the training data, training label, number of epohs to train for, training data was spitted into 50,000 training data and 10,000 test data, and batch size was also declared. Which can be seen in the following code snippet; [5] <br />
 
#### Figure 32: Training the model
Since, I do not have NVIDA graphics card in my computer it took quite a lot of time to train my model and the training results can be seen in the following picture; [5] <br />
  
#### Figure 33: Training result of the model
All the parameters of model fitting are stored in the variable called history, using that variable we can plot the model accuracy when the model was trained. Following codes were used to retrieve the values necessary to plot the model accuracy; <br />
 
#### Figure 34: Model accuracy plot while training
Similarly using the following code we can also plot the model loss; <br />
 
#### Figure 35: Model loss plot while training
Final step of this exercise is to evaluate the model performance using the test dataset, where mean absolute error value was 0.03455 and following code was used; <br />
 
#### Figure 36: Model evaluation.
 <br />
### 3	References
[1] THE MNIST DATABASE. (n.d.). Retrieved June 09, 2017, from http://yann.lecun.com/exdb/mnist/ <br />
[2] Pythonorg. (2017). Pythonorg. Retrieved 9 June, 2017, from https://docs.python.org/3/library/gzip.html <br />
[3] Visualizing MNIST: An Exploration of Dimensionality Reduction. (n.d.). Retrieved June 22, 2017, from http://colah.github.io/posts/2014-10-Visualizing-MNIST/ <br />
[4] Mallick, S. (2017, January 30). Home. Retrieved June 30, 2017, from http://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/  <br />
[5] "Keras Tutorial: The Ultimate Beginner's Guide to Deep Learning in Python." EliteDataScience. N.p., 28 Apr. 2017. Web. 30 June 2017. <br />
[6] Nielsen, Michael A. "Neural Networks and Deep Learning." Neural networks and deep learning. Determination Press, 01 Jan. 1970. Web. 30  <br />

