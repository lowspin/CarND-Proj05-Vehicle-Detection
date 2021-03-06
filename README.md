# CarND-Proj05-Vehicle-Detection

## Background
This repository contains my project report for the [Udacity Self-Driving Car nano-degree](https://www.udacity.com/drive) program's project 5 - Vehicle Detection. The original starting files and instructions can be found [here](https://github.com/udacity/CarND-Vehicle-Detection).
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog_images/hog_car4.jpg
[image2]: ./output_images/hog_images/hog_notcar3.jpg
[image3]: ./output_images/search_windows_all/search_windows_for_all_scales.jpg
[image4]: ./output_images/hotwindows_testimages/hotwindows_0.jpg
[image5]: ./output_images/hotwindows_testimages/hotwindows_1.jpg
[image6]: ./output_images/hotwindows_testimages/hotwindows_2.jpg
[image7]: ./output_images/hotwindows_testimages/hotwindows_3.jpg
[image8]: ./output_images/hotwindows_testimages/hotwindows_4.jpg
[image9]: ./output_images/hotwindows_testimages/hotwindows_5.jpg
[image10]: ./output_images/heatmap_sequence/carpos_heatmap_0.jpg
[image11]: ./output_images/heatmap_sequence/carpos_heatmap_1.jpg
[image12]: ./output_images/heatmap_sequence/carpos_heatmap_2.jpg
[image13]: ./output_images/heatmap_sequence/carpos_heatmap_3.jpg
[image14]: ./output_images/heatmap_sequence/carpos_heatmap_4.jpg
[image15]: ./output_images/heatmap_sequence/carpos_heatmap_5.jpg
[image16]: ./output_images/heatmap_sequence/carpos_heatmap_6.jpg
[image17]: ./output_images/heatmap_sequence/final_heatmap_6.jpg
[image18]: ./output_images/heatmap_sequence/result_6.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! This project is roughly divided into three sections: 
- Extract features using Histogram of Oriented Gradient (HOG), and other features
- Implement a sliding window search to locate matches in single test images, 
- Additional processing for time-sequence video frames, including accumulating detection over multiple frames.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the files `train_spat-hist-hog.py` and `lesson_functions.py`. The main functions are `extract_features()` (`lesson_functions.py` lines 57-105) and `get_hog_features()` (`lesson_functions.py` lines 13-30). The actual heavy lifting of extracting the HOG feawtures is done using skimage's `skimage.feature.hog()` function.  

I've used only the provided training data, which contains both `vehicle` and `non-vehicle` images assembled from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. These data can be found in the original udacity [repo](https://github.com/udacity/CarND-Vehicle-Detection).

Here is an example of HOG features extracted for a `car` image:

![alt text][image1]

Here is an example of HOG features extracted for a `non-car` image:

![alt text][image2]

More samples can be found in this [folder](https://github.com/lowspin/CarND-Proj05-Vehicle-Detection/tree/master/output_images/hog_images/).

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters (`train_spat-hist-hog.py` lines 49-58), in particular, the `color_space` and `hog_channel` parameters. For example I experimented to see if I can get better results from the `saturation` channel of the HSV and HLS color spaces.

The following accuracy scores were observed:

| color_space   | orient        | pix_per_cell  | cell_per_block| hog_channel   | Accuracy      | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| RGB           | 9             | 8             | 2             | ALL           | 97.5%         |
| YCrCb         | 9             | 8             | 2             | ALL           | **98.7%**     |
| HSV           | 9             | 8             | 2             | 1             | 90.3%         |
| HLS           | 9             | 8             | 2             | 2             | 89.3%         |
| YUV           | 9             | 8             | 2             | ALL           | 98.7%         |

In the end, I settled on the following configuration for HOG features:
- color_space = 'YCrCb'
- orient = 9
- pix_per_cell = 8 
- cell_per_block = 2 
- hog_channel = 'ALL' 

In addition, the following parameters are used for the color histogram (`lesson_functions.py` lines 44-52) and spatially binned color (`lesson_functions.py` lines 38-42) features discussed in class:
- spatial_size = (16, 16) 
- hist_bins = 16 
- spatial_feat = True 
- hist_feat = True 
- hog_feat = True 

The combined HOG and color features are stacked and normalized using `sklearn.preprocessing.StandardScaler`'s `fit()` and `transform()` functions (`train_spat-hist-hog.py` lines 75 and 77).

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `sklearn.svm`'s `LinearSVC` object (`train_spat-hist-hog.py` lines 92 and 95). I have applied the standard data shuffling and train/test split of 80:20 using `sklearn.model_selection.train_test_split` (`train_spat-hist-hog.py` lines 84-86). See previous section for the accuracy scores for some configuration settings.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in the `find_cars()` function in the file `genimages.py` (lines 157-226). Instead of overlap, I define how many cells to step in `genimages.py` line 179, `cells_per_step = 2`. I've chosen six different scales to search - 0.5, 1.0, 1.5, 2.0, 2.5, 3.0. Since smaller scale matches only occur further out to the horizon, I further restrict the search location based on the scale, as follows:

| scale         | xstart        | ystart        | xstop         | ystop         | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| 0.5           | 400           | 400           | max           | 450           |
| 1.0           | 350           | 400           | max           | 530           |
| 1.5           | 300           | 400           | max           | 580           |
| 2.0           | 250           | 400           | max           | 600           |
| 2.5           | 200           | 400           | max           | 650           |
| 3.0           | 150           | 400           | max           | 700           |

The overlapping search windows for all scales are plotted below for one of the test image:

![alt text][image3]

The individual images can be found [here](./output_images/search_windows_all/)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on all six scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.  Here are some example images:

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]


As observed, despite efforts to reduce the search area and optimize features, there are some false positive detections. Fortunately, we are working with sequential frames in our test video and we can make detection more reliable by including multiple frames in our detection. This is discussed in the next section.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result file](./result_project_video_frames7.mp4), or view it on [Youtube](https://youtu.be/BxmHD5Y3IbU)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For each window scale, I recorded the positions of positive detections in each frame of the video (`genimages.py` line 264). Results for each window scale are added for each frame (`genimages.py` line 265).

In order to reduce false positives, I implemented a python class called `VehTracker` (`trackers.py` lines 1-30), which is used to hold and combine the hot window positions in the past `nhistory` frames (`genimages.py` lines 268-270). In this way, results from seven frames were added to form the final positions used for detection. Correspondingly, the threshold for detection is increased from 1 to 7 to account for the increased occurences (`genimages.py` line 278).

From the positive detections I created a heatmap (`genimages.py` line 275) and then thresholded that map (`genimages.py` line 278) to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected (`genimages.py` line 286).  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are seven frames and their corresponding heatmaps:

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all seven frames:
![alt text][image17]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image18]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main techniques used in this project were 
1) using HOG plus spatially binned color and histograms of color as the features, 
2) using support vector machine to train and classify labeled dataset using the identified featues and
3) using a sliding window search algorithm to locate probable vehicles.

Each of these steps/techniques has its own issue and limitation:

1) Although the features are proven to work in prior application, I can't help but wonder if other features might work better. We could even use a deep learning approach with convolution network to train and classify the images. A lot of time was spent tuning the feature parameters, but the results are still not fully robust, especially for still images. I notice that some of the false positives are due to areas with high contrast regions, such as shadows and guard rails. To make it more robust, I would experiment with more features, perhaps with different weights.

2) Here I think the choice of support vector machines is actually very appropriate as it is a simple binary classification problem. However, I'm getting some false positives despite the simplicity of the classifier. To make the technique more robust, I would use more training data, especially for non-car category, in order to reduce false positives. I would also train the car category images separately by left, right or straight (back of car) orientation and apply them separate to the left, right and center regions of the camera captured images.

3) The sliding window technique with hot windows heatmaps is a logical approach, however a lot depends on the choice of scale, overlap and region to search. These also changes with the curvature of the road and other conditions, such as if we're travelling on a one- or two-way street. Hence, to make it more robust, I would experiment with more window settings, including configuratin for multiple frame accumulation and thresholds.

