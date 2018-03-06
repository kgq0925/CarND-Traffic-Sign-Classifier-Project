# Traffic Sign Recognition

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the German Traffic Sign dataset.
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[samples]: ./examples/samples.png "RANDOM SAMPLES"
[image1]: ./new-images/small_120_kmh_limit.png "RANDOM SAMPLES"
[image2]: ./new-images/small_keep_right.png "RANDOM SAMPLES"
[image3]: ./new-images/small_no_vehicles.png "RANDOM SAMPLES"
[image4]: ./new-images/small_priority_road.png "RANDOM SAMPLES"
[image5]: ./new-images/small_road_works.png "RANDOM SAMPLES"

### Data Set Summary & Exploration

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

![alt text][samples]

### Design and Test a Model Architecture

#### Pre-process

1. All images are converted to YUV space. The Y channel is then preprocessed with global and local contrast normalization while U and V channels are left unchanged.
2. Samples are randomly perturbed in position, in scale and rotation. When a dataset does not naturally contain those deformations, adding them synthetically will yield more robust learning to potential deformations in the test set

#### Model Architecture

My final model consisted of the following layers:

| Layer					| Description									|
|:---------------------:|:---------------------------------------------:|
| Input					| 32x32x3 RGB image								|
| Convolution 3x3		| 1x1 stride, same padding, outputs 32x32x64	|
| RELU					|												|
| Max pooling			| 2x2 stride, outputs 16x16x64					|
| Convolution 3x3		| 1x1 stride, same padding, outputs 32x32x128	|
| RELU					|												|
| Max pooling			| 2x2 stride, outputs 16x16x128					|
| Convolution 3x3		| 1x1 stride, same padding, outputs 32x32x38	|
| RELU					|												|
| Max pooling			| 2x2 stride, outputs 16x16x38					|
| Fully connected		|												|
| Output				|												|

 
#### Train, Validate and Test the Model

| Parameter			| Value	|
|:-----------------:|:-----:|
| LEARNING RATE		| 0.001	|
| EPOCHS			| 50	|
| BATCHES_PER_EPOCH	| 10000	|
| BATCH_SIZE		| 128	|

To train the model, I used an "tf.train.AdamOptimizer"

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.95

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

Here are the results of the prediction:

| Image					| Prediction									|
|:---------------------:|:---------------------------------------------:|
| Speed limit (120km/h)	| Speed limit (50km/h)							|
| Keep right			| Keep right									|
| No vehicles			| Speed limit (50km/h)							|
| Priority road			| Priority road					 				|
| Road work				| Road work										|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.


For the first image, the top five soft max probabilities were

| Probability			| Prediction									|
|:---------------------:|:---------------------------------------------:|
| 0.54					| Speed limit (50km/h)							|
| 0.20					| Speed limit (30km/h)							|
| 0.19					| Speed limit (70km/h)							|
| 0.03					| Speed limit (100km/h)							|
| 0.01					| Speed limit (20km/h)							|

For the second image

| Probability			| Prediction									|
|:---------------------:|:---------------------------------------------:|
| 0.99					| Keep right									|
| 0.0009				| Turn left ahead								|
| 0.00001				| Speed limit (50km/h)							|
| 0.000007				| Speed limit (80km/h)							|
| 0.000003				| No passing for vehicles over 3.5 metric tons	|

For the third image

| Probability			| Prediction									|
|:---------------------:|:---------------------------------------------:|
| 0.71					| Speed limit (50km/h)							|
| 0.11					| Speed limit (80km/h)							|
| 0.04					| Speed limit (100km/h)							|
| 0.03					| Speed limit (120km/h)							|
| 0.02					| Speed limit (60km/h)							|

For the fourth image

| Probability			| Prediction									|
|:---------------------:|:---------------------------------------------:|
| 0.71					| Priority road									|
| 0.19					| Roundabout mandatory							|
| 0.06					| Speed limit (100km/h)							|
| 0.007					| Speed limit (60km/h)							|
| 0.004					| Speed limit (80km/h)							|

For the fifth image

| Probability			| Prediction									|
|:---------------------:|:---------------------------------------------:|
| 0.99					| Road work										|
| 0.001					| Beware of ice/snow							|
| 0.001					| Road narrows on the right						|
| 0.001					| Children crossing								|
| 0.0004				| Bumpy road									|
