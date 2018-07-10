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
[image8]: ./report/german_traffic_signs.png "German Traffic Signs"
[image9]: ./report/softmax_probabilities_traffic_signs.png "Softmax Probabilities Traffic Signs"
[image9_02]: ./report/softmax_probabilities_traffic_signs_02.png "Softmax Probabilities Traffic Signs"
[image9_03]: ./report/softmax_probabilities_traffic_signs_03.png "Softmax Probabilities Traffic Signs"
[image9_04]: ./report/softmax_probabilities_traffic_signs_04.png "Softmax Probabilities Traffic Signs"
[image9_05]: ./report/softmax_probabilities_traffic_signs_05.png "Softmax Probabilities Traffic Signs"
[image9_06]: ./report/softmax_probabilities_traffic_signs_06.png "Softmax Probabilities Traffic Signs"
[image9_07]: ./report/softmax_probabilities_traffic_signs_07.png "Softmax Probabilities Traffic Signs"
[image9_08]: ./report/softmax_probabilities_traffic_signs_08.png "Softmax Probabilities Traffic Signs"
[image9_09]: ./report/softmax_probabilities_traffic_signs_09.png "Softmax Probabilities Traffic Signs"
[image9_10]: ./report/softmax_probabilities_traffic_signs_10.png "Softmax Probabilities Traffic Signs"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

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
| Classifier Layer 2: Network	    | 43 Hidden Units       									|									|
| Softmax cross entropy with logits				|         									|
| Loss operation						| reduce mean												|
| Optimizer						| AdamOptimizer learning rate 0.0002												|
 

### Model Training

#### Layout
I started with the LeNet model given in the excersices, it it showed already good results to start with.

| Epoch | Batch Size | Learn Rate | Normalize | Arch | Picture | Layer 1 | Layer 2 | Classifier L1 | Classifier L2 | Classifier L3 | Accuracy Validation |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|30|32|0,0005|Global|SS|Y Channel|6|16|120|84|43|0.944|
|30|32|0,0005|Global|SS|Gray scale|6|16|120|84|43|0.939|

The goal was to see the effect of simple gray scaling the traffic signs versus using the luma channel. As the luma channel showed better performance I worked with that.

In a next attempt I changed the layout of the network similar to the recommendation from Pierre Sermanet and Yann LeCun.  This impoved the performance way more as you can see in the next table.

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

This impoves the contrast of edges a lot, even if pictures histogram looks already widespread. This does not only help the human eye to recognize a picture but is also benefitial for the network as features are exposed.

#### Dropout

Learning without dropout looks like this. The accuracy increases fast but gets never stable. Lowering the learning rate didn't smooth the graph as desired. 

![alt text][image5]

Indroducing dropout after convolution and between the two classification layers smoothed the learning and gave again another performance boost. 

![alt text][image6]

| Epoch | Batch Size | Learn Rate | Normalize | Arch | Picture | Layer 1 | Layer 2 | Classifier L1 | Classifier L2 | Accuracy Validation |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|30|32|0,0002|Local + Global|SS|Y Channel|108|200|57|43|0.994|

** Testing Accuracy: 0.984 **

#### MS versus SS Architecture

At last, I have implemented the MS architecture described by Pierre Sermanet and Yann LeCun. 

![alt text][image7]

| Epoch | Batch Size | Learn Rate | Normalize | Arch | Picture | Layer 1 | Layer 2 | Classifier L1 | Classifier L2 | Accuracy Validation |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|30|32|0,0002|Local + Global|MS|Y Channel|108|108|57|43|0.993|

** Testing Accuracy: 0.972 **

Unlike the experience from Pierre Sermanet and Yann LeCun I could not see at first sight that the MS architectue improves the performance of the network. I have choosen snow covered traffic signs for a final test. As those were better recognized by the MS architecture, I decided to choose this MS even though the test accuracy is lower. I belive the SS architecture was already overfitting.

#### Final Results
My final model results were:

* validation set accuracy of 0.993 
* test set accuracy of 0.972
 
### Test a Model on New Images

Here are ten German traffic signs that I found on the web:

![alt text][image8]

I have choosen also snow covered, rotated and low contrast traffic signs to see if the network is able to classify them properly.

#### Result

The model was able to classified all traffic signs correctly and was in all cases 100 percent sure.  Even with the partly visible snow covered one. 

Please see the softmax probabilities in the following picture. 

![alt text][image9]
![alt text][image9_02]
![alt text][image9_03]
![alt text][image9_04]
![alt text][image9_05]
![alt text][image9_06]
![alt text][image9_07]
![alt text][image9_08]
![alt text][image9_09]
![alt text][image9_10]

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook.