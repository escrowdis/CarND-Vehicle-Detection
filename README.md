# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[hog_v]: ./imgs/hog_vehicle.png "HoG Extracted Image with Vehicle"
[imgs_v]: ./imgs/imgs_vehicle.png "Image with Vehicle"
[hog_nv]: ./imgs/hog_nonvehicle.png "HoG Extracted Image with non-Vehicle"
[imgs_nv]: ./imgs/imgs_nonvehicle.png "Image with non-Vehicle"
[img_t_0]: ./imgs/t_0.png "Result of Vehicle Detection with Heat Map - 0"
[img_t_1]: ./imgs/t_1.png "Result of Vehicle Detection with Heat Map - 1"
[img_t_2]: ./imgs/t_2.png "Result of Vehicle Detection with Heat Map - 2"
[img_t_3]: ./imgs/t_3.png "Result of Vehicle Detection with Heat Map - 3"
[img_t_4]: ./imgs/t_4.png "Result of Vehicle Detection with Heat Map - 4"
[img_t_5]: ./imgs/t_5.png "Result of Vehicle Detection with Heat Map - 5"
[data_comb]: ./imgs/data_comb.png ""
[data_comb_norm]: ./imgs/data_comb_norm.png ""
[img_tc_0]: ./imgs/tc_0.png "Result of Vehicle Detection with Heat Map - 0"
[img_tc_1]: ./imgs/tc_1.png "Result of Vehicle Detection with Heat Map - 1"
[img_tc_2]: ./imgs/tc_2.png "Result of Vehicle Detection with Heat Map - 2"
[img_tc_3]: ./imgs/tc_3.png "Result of Vehicle Detection with Heat Map - 3"
[img_tc_4]: ./imgs/tc_4.png "Result of Vehicle Detection with Heat Map - 4"
[img_tc_5]: ./imgs/tc_5.png "Result of Vehicle Detection with Heat Map - 5"
[img_video_0]: ./imgs/video_0.png "Video cap 0"
[img_video_1]: ./imgs/video_1.png "Video cap 1"

<!-- [img_hsl]: ./imgs/img_hsl.png "" -->

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

### Dataset

First of all, the datasets were loaded, excluding the one from Udacity due to limited time. After that, the model was trained by the processed data which spent the most of time to do it every time because it took a lot to compute HoG. There are total 8792 and 8968 images in vehicle and non-vehicle sets, respectively. The dataset was then divided into training and test set with split ratio 0.2:

- Training dataset: 14208
- Test dataset: 3552

There's random samples for visualization of two sets, vehicle and non-vehicle. Images with vehicle have a kind of circular shape in HoG image, which the one without vehicle type barely have.

| Vehicle |
|---------|
|![][imgs_v]|
|![][hog_v]|

|Non-vehicle|
|---------|
|![][imgs_nv]|
|![][hog_nv]|

### Algorithm Implementation
Here's the parameters applied in the FeatureExtraction function listed below. I remembered the paramters listed below is commonly used in HoG, that's why I decided to use these values as default. The HSV color space was chosen again which Hue and Saturation are critical one, because vehicle is a block in the image with similar color which might be easy to detect. The parameters used for histogram was the same as the lesson. The Multi-scale windows for searching was applied due to the perspective of the image. Closer object stands for more pixels in the image, otherwise, the farer one stands for less pixels; Furthermore, to exclude some region absolutely don't have car in there like sky. So~the feature extraction implementation was almost done. After all the features were extracted, they are combined into a single array as one feature. The code below is applied to normalize the feature that each of them may have different range shown below. After that, the total length of a single feature vector is 2628.

```python
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X_train and X_test
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)
```


|Before|After Normalization|
|-|-|
|![][data_comb]|![][data_comb_norm]|

**For HoG**
- orient = 9
- pixels per cell = 8
- cells per block = 2
- channel = 0 (Hue channel only)

**For Spatial Binning**
- color: 'HSV'
- size = (16, 16)

**For Histogram of Color**
- bins = 32
- bins_range = (0, 256)

**Multi scale Windows for Searching**
| X range   | Y range  | Windows size |
|-          |-         |-             |
|[0, 680]   |[400, 600]|(64, 64)      |
|[600, 1280]|[400, 650]|(64, 64)      |
|[480, 800] |[400, 560]|(128, 128)    |

Since the feature extraction is done. The linear SVM was chosen as the training model.

```python
# Use a linear SVC
svc = LinearSVC()
svc.fit(X_train, y_train)
```

The recognition rate is 97.33% which looks not bad. After that, I test it on test_images with also good recognition, but still has some false positive. During the implementation of multi-scale windows, I made a mistake to cast windows with wrong type which cost me almost a whole afternoon to debug on this stuffs. The result showed not proper not matter how I adjusted my algorithm, I even copied the lesson's code to validate.

**7.65 Seconds to train SVC...
Test Accuracy of SVC =  0.9761
My SVC predicts:	     [0. 0. 0. 1. 1. 1. 1. 0. 0. 0.]
For these 10 labels:	 [0. 0. 0. 1. 1. 1. 1. 0. 0. 0.]
0.00148 Seconds to predict 10 labels with SVC.**

### Results

Here's the result from the images in 'test images' folder. From the heat map you can see there are several boxes detected on the white vehicle. Due to one object has only one window. The 'add_heat' function was applied to merge all the boxes into one using the method similar to non maximum suppression.

![][img_t_0]
![][img_t_1]
![][img_t_2]
![][img_t_3]
![][img_t_4]
![][img_t_5]

![][img_tc_0]
![][img_tc_1]
![][img_tc_2]
![][img_tc_3]
![][img_tc_4]
![][img_tc_5]

The images shown below is the best detection in my video. The size of rectangles may vary sometimes, I think it can be solved by averaging history to calculate a smoother one.

![][img_video_0]
![][img_video_1]

#### Videos

In video 1, I set the window size for multi scale searching to (128, 128). The noise might be filtered. But it comes with a shortcoming that if the object were too far away from us, the represented pixels would be too less to be detected by the algorithm. Otherwise, in the video 2, the window size for multi scale searching is set to (90, 90). This caused some wrong rectangles popped up and then disappeared rapidly due to mismatching(?). There's also a short video (Video 3), test_video.mp4' in the folder. It also can detect the vehicles but with some noises. I think this can be fixed by setting each rectangle a lives, it can be detected as an object if the lives is higher than a threshold by temporal. this will be on the list for next improvement!

[![](https://img.youtube.com/vi/VCd33sKjt50/0.jpg)](https://youtu.be/VCd33sKjt50)
[Link of Video 1](https://youtu.be/VCd33sKjt50)

[![](https://img.youtube.com/vi/Q2DfxsQEOkE/0.jpg)](https://youtu.be/Q2DfxsQEOkE)
[Link of Video 2](https://youtu.be/Q2DfxsQEOkE)

[![](https://img.youtube.com/vi/5qd3vWtqs_I/0.jpg)](https://youtu.be/5qd3vWtqs_I)
[Link of Video 3](https://youtu.be/5qd3vWtqs_I)

### Discussion
In this project, I've to admitted the time is too rush for me to enhance the performance. There are two shortcomings in current algorithm:
- False positive too high
- Failed to continuous track the objects

You can see the left side gap were sometimes detected as vehicle, but just a flash time not always there. This can be solved by add the lives method to check how long have the rectangles been detected. If the rectangle has been detected for 3 frames (threshold), then it was considered as an object in this region. On the other side, the object was failed to track continuously due to algorithm limitations. This can be fixed by assuming an object will not disappear or show up suddenly. The predicted rectangle will be displayed if the previous condition has been satisfied, but there's no object detected in region around it this time.
