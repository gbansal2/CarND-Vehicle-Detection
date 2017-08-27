**Vehicle Detection Project**

[//]: # (Image References)
[image1]: ./vehicles/GTI_MiddleClose/image0000.png
[image2]: ./non-vehicles/GTI/image59.png
[image3]: ./car_hog_Schannel.png
[image4]: ./notcar_hog_Schannel.png
[image5]: ./output_images/allboxes_test4.jpg
[image6]: ./output_images/detections_test4.jpg
[image7]: ./output_images/detections_test6.jpg
[image8]: ./output_images/detections_test3.jpg
[image9]: ./output_images/heatmap_test4.jpg
[video1]: ./output_videos/project_video_tracking.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

This document is the writeup for this project with details on algorithms chosen and implementations.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines #211 through #263 of the file called `helper_functions.py`.  Also, the actual call to the hog function from the skimage library is in lines #6 to #23 of the same file.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt_text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example for a car and a not-car image, using the `HLS` color space and using only the 'S' channel, and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(4, 4)`:


![alt text][image3]
![alt_text][image4]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters for HOG features, such as orientations, pixels per cell, cells per block, color spaces, etc. I changed these parameters to increase the feature vector size and monitored the trainng error. I settled on the set of parameters for which I obtained >98% training accuracy.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the training data set. The code for training a classifier is in file `train_model.py` from lines #85 to #101. The training dataset consists of hog features, color features, and spatial features. The dataset is randomly split into a training set and a validation set (20%). I obtained a validation set accuracy of greater than 98%. 

The model was saved as a pickle file which can be read later during the inference stage. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented a sliding window search in the function `find_cars` in `helper_functions.py` file. I generated windows of three sizes, with sizes 64 x 64, 96 x 96, and 128 x 128. The application range for each window size is different. The smallest window is applied towards the middle of the image, and the largest window is applied towards the bottom. This is because the vehicles are bigger in size, the closer they are to the camera.
The hog features are extracted once for the entire image, and are filtered as the window is slided across the image.

The figure below shows all the windows for an example image.

![alt text][image5]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt_text][image7]
![alt_text][image8]

Note that in the bottom image, there is a false detection. This is handled using a heatmap method as described in the Video Implementation section below.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video_tracking.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  
To reduce false positives, I created a heatmap of positive detections and averaged the value of heatmap over 5 successive video frames. I used a Tracker Class to store the heatmaps. The code for this is implemented in lines #12 to #15 and #43 to #72 of file `apply_image.py`. 

I then thresholded that map to identify vehicle positions, and then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Here's an example result showing the heatmap from an example image:

![alt text][image9]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

