## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

[//]: # (Image References)
[image1]: ./examples/video1_40_x.png
[image2]: ./examples/video1_40_cars.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to design a convolution neural network 

My first step was to choose from several techniques in Semantic segmentation and i choose full convolution neural network model which is very popular in this kind of problem.
 
In order to gauge how well the model was working, I used the normal and the segmanted images as a training set.I use very samll data set for training which cause overfitting.

To combat the overfitting, I modified the model so that increase the depth of the model to reach 7 convolution layers then used after each intermediate convolution layer a normalization and a dropout layers. 

This approach provided me with good enough model that can highlight cars and to overcome the having a huge size of network i preprocess the data and crop the input image to be as small as possible and the output layer as a gray scale image.

The final step was to run the pipline to see how well the car was dedected on the road. There were a few spots where the vehicle fell off the track so to improve the dedection behavior in these cases, I increased training batches and learning rate.

At the end of the process, the vehicles was dedected on the road.

#### 2. Final Model Architecture

The final model architecture (Car)Model.py lines 65-94) consisted of a convolution neural network with the following layers and layer sizes

1- Convolution layer with filter size = 20, kernel_size= ( 5, 5) and same border mode
2- Batch normalization then relu activation
3- Convolution layer with filter size = 30, kernel_size= ( 5, 5) and same border mode
4- Batch normalization then relu activation
5- Convolution layer with filter size = 30, kernel_size= ( 5, 5) and same border mode
6- Batch normalization then relu activation
7- Convolution layer with filter size = 30, kernel_size= ( 5, 5) and same border mode
8- Batch normalization then relu activation
9- Convolution layer with filter size = 20, kernel_size= ( 5, 5) and same border mode
10- Batch normalization then relu activation
11- Convolution layer with filter size = 10, kernel_size= ( 5, 5) and same border mode
12- Batch normalization then relu activation
13- Convolution layer with filter size = 10, kernel_size= ( 5, 5) and same border mode
14- L2 regularizer and relu activation
15- compiled using loss Function 'mean_squared_error' and an adam optimizer

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior,

I first got the training data from Eric Lavigne on slacks.

the data set contain of 14 images from the original video and processed to get the location of the car and its shape by highlighting it.

![alt text][image1]
![alt text][image2]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.

I tried to use a gray scale image as an input to my FCN  but it gave vad results comparing to RGB images with the same parameter. I think that drop of perfomance is due to giving away almost all the usefull information considering color space.

I think if i used larger data set i could cover more corner cases.

