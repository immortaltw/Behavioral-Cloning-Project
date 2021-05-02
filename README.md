# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./resources/cnn-architecture-nvidia.png "Model Visualization"
[image2]: ./resources/center.jpg "Center"


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* video.mp4 showing the result of autonomous run using trained model
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I use a convolution neural network model similar to Lenet (line 126)as starting point. It's relatively simple and fast to train. However, the model was only good enough to make the car run for 1/2 lap in autonomous mode.

I split my image and steering angle data into a training and validation set with 80/20 ratio. In the beginning I set the epochs to 7. But 7 seemed too big because the validation error began to fluctuate after epoch 5. So I set the epochs to 3 for this model.

#### 2. Final Model Architecture

Because the Lenet model was not good enough, I tried to use Nvidia's model but with some slight difference (line 94). The model I used have 5 CNN layers with kernal sizes identical to Nvidia's network arch. I added one max pooling layer after the first CNN, and another max pooling and a dropout (0.25).

Unlike Nivdia's model, which has 4 fully connected layer, I add a dropout layer after the first fully connected layer. I also added 3 max pooling layer in between. The validation error of this model began to fluctuate after epoch 4, so I set epochs to 3 to avoid overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, basically the region after crossing the bridge. To improve it I simply record 2 more runs specifically for this region and use it as part of training data set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Here is a visualization of the architecture (Source: Nvidia)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 4 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover itself if unfortunately it drives itself to the curb.

Then I capture 2 laps of driving record on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would teach the model to learn how to drive on the opposite direction, since the default driving direction of the track is CCW.

I also recored 3 more data sets specifically for the curve after the bridge to further augment the learning for driving through this curve.

After the collection process, I had 16070 data points. I then preprocessed this data by first normalizing the images, then crop the region 75 pixels from the top of the images and 25 pixels from the bottom of the images.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by observing the validation error per epoch during training. I used an adam optimizer with learning rate 0.001 and 0.0001 decay value.

As a side note, I tried to use generator but it made the training extremely slow. It was way too slow so I gave up.
