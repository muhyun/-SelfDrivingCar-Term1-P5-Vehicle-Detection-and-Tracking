[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image8]: ./images/sample-pipeline-output.png
[image9]: ./images/heatmap.png
[image10]: ./images/final-image.png
[video1]: ./project_video.mp4

---


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook.

* get_hog_features() : to calculate HOG features
* bin_spatial() : to calculate spatial features
* color_hist() : to calculate histogram of the color channels

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I selected parameters as;

| Parameter Name | Value |
|----------------|-------|
| Color Space | 'YCrCb' |
| HOG orientations |  9  | 
| HOG pixels per cell | 8 |
| HOG cells per block | 2 |
| Hog Channel | ALL |
|Spatial binning dimensions | (32,32) |
| Number of histogram bins | 32 |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using training data which has 8792 cars and 8968 non-cars. These images are in png format. I also applied image augmentation by flipping images vertically to have more training data. Using the default hyper-parameters, SVC give a reliable model for vehicle detecting.

```python
svc = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1,loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2',random_state=None, tol=0.0001, verbose=0)
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First, I used the basic window search method suggested by Udacity class material, but I came up with the efficient implementation verions which also are suggested by Udacity class. The code for this is found in the jupyter notebook, _vehicle-detection-and-tracking.ipynb_. The name of the function for this is _find_cars_.

I modified the original function to use step 1 for all scale except 3 for 0.74 scale. Also, I restrict the search window area to (500, 1280) in X-axis.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales (0.75, 1, 1.5, and 2) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![alt text][image8]

---

### Video Implementation

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video-result.mp4)

[![Alt text for your video](https://img.youtube.com/vi/T-D1KVIuvjA/0.jpg)](https://youtu.be/iQEmfnoFxZI)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are 2 frames and their corresponding heatmaps:

![alt text][image9]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image10]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project, a traditional machine learning model, SVC, is used. But I coudl apply a deep learning based object detection algorithm as more robust vehicle detection model.

