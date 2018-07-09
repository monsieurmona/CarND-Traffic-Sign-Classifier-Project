# **Traffic Sign Recognition** 

## Writeup - Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

This is a link to my [project code](https://github.com/monsieurmona/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

I have used the experiences described in paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" written by Pierre Sermanet and Yann LeCun. This helped a lot to get this exercise done. Nevertheless, I have conducted my own experiements as it can be seen below.

[//]: # (Image References)

[image1]: ./report/distribution_of_training_data.png "Distribution of traning data"
[image2]: ./report/example_traffic_signs.png "Visualization"
[image3]: ./report/luma_channel_local_contrast.png "Local Contrast of Luma Channel"
[image4]: ./report/perturbed_images.png "Random Noise"
[image5]: ./report/learning_without_dropout.png "Learning Without Drop Out"
[image6]: ./report/learning_with_dropout.png "Learning With Drop Out"
[image7]: ./report/ms_architecture.png "MS Architecture"
[image8]: ./examples/placeholder.png "Traffic Sign 3"
[image9]: ./examples/placeholder.png "Traffic Sign 4"
[image10]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation. |30|32|0,0005|Global|SS|Y Channel|108|200|57|43| |0.958|


### Data Set Summary & Exploration

The summary is calculated using python with the help of numpy.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

The bar shows the distribution of traffic sign in the training set.

![alt text][image1]

### Design and Test a Model Architecture

The original training data looks like this. 

![alt text][image2]

As a first step, I decided to convert the images to grayscale as Pierre Sermanet and Yann LeCun wrote in their paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" that color doesn't contribute to the learning of traffic signs. Like them I have choosen to use the luma channel from YUV color space. 

This channel is then processed first by local normalization and then by global normalization. Local normalization increases the contrast within the image, whereas global normalization centers all images around its mean. 

Here are examples after grayscaling and normalization.

![alt text][image3]

To increase the variance of the dataset I added four additional images for each training image. The processing pipeline is like this:

* translate randomly by [-2, 2] pixel in any direction
* rotate by random degree [-15, 15] 
* scale in any direction by pixels in range [-2, 2]

![alt text][image4]

### Final model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:--------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Layer 1: Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x108 	|
| Layer 1: RELU					|												|
| Layer 1: Max pooling	      	| 2x2 stride,  outputs 14x14x108 				|
| Layer 2: Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x108 	|
| Layer 2: RELU					|												|
| Layer 2: Max pooling	      	| 2x2 stride,  outputs 5x5x108 				|
| Dropout Layer 1	      	| Keep Propability: 0.5 				|
| Dropout Layer 2	      	| Keep Propability: 0.5 				|
| Flatten and Combine	      	| outputs 57153600 x 1 				|
| Fully connected| | 
| Classifier Layer 1: Network	    | 57 Hidden Units       									|
| Classifier Layer 1: RELU		| 												|
| Classifier Layer 1: Dropout	      	| Keep Propability: 0.5 				|
| Classifier Layer 1: Network	    | 43 Hidden Units       									|
| Fully connected		| etc.        									|
| Softmax cross entropy with logits				|         									|
| Loss operation						| reduce mean												|
| Optimizer						| AdamOptimizer learning rate 0.0002												|
 

### Model Training

#### Layout
I started with the LeNet model given in the excersices:

| Epoch | Batch Size | Learn Rate | Normalize | Arch | Picture | Layer 1 | Layer 2 | Classifier L1 | Classifier L2 | Classifier L3 | Accuracy Validation |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|30|32|0,0005|Global|SS|Y Channel|6|16|120|84|43|0.944|
|30|32|0,0005|Global|SS|Gray scale|6|16|120|84|43|0.939|

The goal was to see the effect of simple gray scaling the traffic signs versus using the luma channel. As the luma channel showed better performance I worked with that.

In a next attempt I changed the layout of the network. This impoved the performance way more as you can see in the next table.

| Epoch | Batch Size | Learn Rate | Normalize | Arch | Picture | Layer 1 | Layer 2 | Classifier L1 | Classifier L2 | Classifier L3 | Accuracy Validation |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|30|32|0,0005|Global|SS|Y Channel|6|16|120|84|43|0.944|
|30|32|0,0005|Global|SS|Y Channel|108|200|43| | |0.957|
|30|32|0,0005|Global|SS|Y Channel|108|200|57|43| |0.958|

#### Local Normalization

Another performance gain gave local normalization of training images.

| Epoch | Batch Size | Learn Rate | Normalize | Arch | Picture | Layer 1 | Layer 2 | Classifier L1 | Classifier L2 | Accuracy Validation |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|30|32|0,0005|Global|SS|Y Channel|108|200|57|43|0.958|
|30|32|0,0005|Local + Global|SS|Y Channel|108|200|57|43|0.983|

#### Dropout

Learning without dropout looks like this. The accuracy increases fast but gets never stable. Lowering the learning rate didn't smooth the graph as desired. 

![alt text][image5]

Indroducing dropout after convolution and in between the two classification layers smoothed the learning and gave again another performance boost. 

![alt text][image6]

| Epoch | Batch Size | Learn Rate | Normalize | Arch | Picture | Layer 1 | Layer 2 | Classifier L1 | Classifier L2 | Accuracy Validation |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|30|32|0,0002|Local + Global|SS|Y Channel|108|200|57|43|0.994|

** Testing Accuracy: 0.984 **

#### MS versus SS Architecture

Finally I have implemented the MS architecture described by Pierre Sermanet and Yann LeCun. 

![alt text][image7]

| Epoch | Batch Size | Learn Rate | Normalize | Arch | Picture | Layer 1 | Layer 2 | Classifier L1 | Classifier L2 | Accuracy Validation |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|30|32|0,0002|Local + Global|MS|Y Channel|108|108|57|43|0.993|

** Testing Accuracy: 0.972 **

Unlike the experience from Pierre Sermanet and Yann LeCun I could not see on the first sight that this improves the performance of the network. I have choosen snow covered traffic signs for a final test. As those were better recognized by the MS architecture, I decided to choose this MS even though the test accuracy is lower. 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


