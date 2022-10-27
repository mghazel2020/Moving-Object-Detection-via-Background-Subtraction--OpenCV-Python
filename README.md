# Moving Object Detection using Background-Subtraction in OpenCV-Python

<img src="images/banner-03.jpg" width="1000"/>

## 1. Objective

The objective of this project is to demonstrate object detection using background subtraction OpenCV-Python built-in functionalities. 
We illustrate the development process step by step and present sample detection results.

## 2. Background Subtraction

The objective of this project is to demonstrate object detection using background subtraction OpenCV-Python built-in functionalities. 
We illustrate the development process step by step and illustrate the intermediate and the final detection results.

### 2.1. Background Subtraction Algorithms.

Background subtraction is a simple approach for estimating and segmenting the background from image and then extracting it in order to 
detect potential potential foreground changes, due to motion or introduction of new objects to the imaged scene. One key assumption in 
background subtraction is that imaging camera system if fixed so that changes in the imaged scene are due to changes in the foreground 
and not displacement of the camera. 

Many fixed-camera real-world application make use of simple approaches, such as background subtraction to detect and track changes in 
the imaged scene. For example, in security camera applications, background subtraction can be used to detect people, vehicles and other 
objects of interest entering the imaged scene. The detection output of the background subtraction system can then fed into more 
sophisticated computer systems to analyze the detected objects of interest, such as tracking, counting, or recognizing them. 

There several variations of background subtraction algorithms in the literature and OpenCV has implemented three such algorithms,  
which are very easy to use through C++ and Python API. In this section, we shall briefly outline these four background subtraction algorithms.

#### 2.1.1 BackgroundSubtractorMOG

BackgroundSubtractorMOG is a Gaussian Mixture-based background vs. foreground segmentation algorithm. It models each background pixel 
using a mixture of K Gaussian distributions (K = 3 to 5). The weights of the mixture represent the time proportions that those colors 
stay in the scene. The probable background colors are the ones which stay longer and more static.

#### 2.1.2  BackgroundSubtractorMOG2

This is an improved version of the original BackgroundSubtractorMOG algorithm. One important feature of this algorithm is that it selects the 
appropriate number of gaussian distribution for each pixel, unlike the case for the original BackgroundSubtractorMOG2 algorithm which a  
fixed K gaussian distributions throughout the image. Thus, this new version of the algorithm provides better adaptability to artificial 
variations in scenes, such as those due to changes in illumination changes, shadow, etc. 


#### 2.1.3  BackgroundSubtractorGMG

This algorithm combines statistical background image estimation and per-pixel Bayesian segmentation. It uses first few (120 by default) 
frames for background modelling. It employs probabilistic foreground segmentation algorithm that identifies possible foreground objects 
using Bayesian inference. The estimates are adaptive; newer observations are more heavily weighted than old observations to accommodate 
variable illumination. Several morphological filtering operations like closing and opening are done to remove unwanted noise. Since the 
first 100 or frames are sued for modeling and estimating the background, this algorithm does not  yield reliable detection results for 
these frames. Thus, it is recommended to only start computing the change detection results after skipping 100 to 200 frames when the 
background has been estimated sufficiently well. 


#### 2.1.4 BackgroundSubtractorKNN()

This algorithm is based on the K-nearest neighbors background vs. foreground segmentation and classification algorithm. 
It was shown this this algorithm may be quite efficient when number of foreground pixels is low. Thus this algorithm may be 
more suitable for detecting small changes and objects.


## 3. Data

* We shall use the following OpenCV sample date video:

    * File: vtext.avi
    * Frame size: 575x768 pixels (RGB)
    * Number of frames = 795
    * Duration: 1 min 19 secs
    * Frame rate: 10frames/sec.

<img src="images/vtext-avi-sample-frame.PNG" width="1000"/>

## 4. Development

* In this section, we shall develop the background-subtraction algorithms using OpenCV Python and illustrate the intermediate and final result:

  * Author: Mohsen Ghazel (mghazel)
  * Date: March 29th, 2021
  * Project: Object Detection via Background Subtraction

* The objective of this project is to demonstrate change and object detection and localization via background-subtraction using OpenCV-Python built-in functionalities:
  * Background subtraction is a way of estimating and eliminating the background from image.
* Changes are detected by extracting the moving foreground from the static background.


### 4.1. Step 1: Imports and global variables


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Python imports and environment setup</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># opencv</span>
<span style="color:#800000; font-weight:bold; ">import</span> cv2
<span style="color:#696969; "># numpy</span>
<span style="color:#800000; font-weight:bold; ">import</span> numpy <span style="color:#800000; font-weight:bold; ">as</span> np
<span style="color:#696969; "># matplotlib</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>pyplot <span style="color:#800000; font-weight:bold; ">as</span> plt
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>image <span style="color:#800000; font-weight:bold; ">as</span> mpimg

<span style="color:#696969; "># input/output OS</span>
<span style="color:#800000; font-weight:bold; ">import</span> os 

<span style="color:#696969; "># date-time to show date and time</span>
<span style="color:#800000; font-weight:bold; ">import</span> datetime

<span style="color:#696969; "># Use %matplotlib notebook to get zoom-able &amp; resize-able notebook. </span>
<span style="color:#696969; "># - This is the best for quick tests where you need to work interactively.</span>
<span style="color:#44aadd; ">%</span>matplotlib notebook

<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Global variables</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># OpenCV offers 4 background-subtraction algorithms:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; ">#  Method 1: BackgroundSubtractorMOG:</span>
<span style="color:#696969; ">#     - It is a Gaussian Mixture-based </span>
<span style="color:#696969; ">#       Background/Foreground Segmentation Algorithm.</span>
<span style="color:#696969; ">#  Method 2: BackgroundSubtractorMOG2:</span>
<span style="color:#696969; ">#     - It is also a Gaussian Mixture-based </span>
<span style="color:#696969; ">#       Background/Foreground Segmentation Algorithm. </span>
<span style="color:#696969; ">#     - It provides better adaptability to varying </span>
<span style="color:#696969; ">#       scenes due illumination changes etc.</span>
<span style="color:#696969; ">#  Method 3: BackgroundSubtractorGMG: </span>
<span style="color:#696969; ">#      - This algorithm combines statistical background </span>
<span style="color:#696969; ">#        image estimation and per-pixel Bayesian segmentation.</span>
<span style="color:#696969; ">#  Method 4: BackgroundSubtractorKNN: </span>
<span style="color:#696969; ">#      - This algorithm is K-nearest neighbors </span>
<span style="color:#696969; ">#        clustering and classification.</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Select the OpenCV background-subtraction method that </span>
<span style="color:#696969; "># will be applied</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
OPENCV_BACKGROUND_SUBTRACTION_METHOD <span style="color:#808030; ">=</span> <span style="color:#008c00; ">1</span>

<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Background subtraction is susceptible to noise:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; ">#  - This is due to artifitial changes due to </span>
<span style="color:#696969; ">#    illumination</span>
<span style="color:#696969; ">#  - These sperious changes tend to be small and can </span>
<span style="color:#696969; ">#    be filtered out using an area threshold</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
MIN_CONTOUR_AREA <span style="color:#808030; ">=</span> <span style="color:#008c00; ">100</span>

<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Background subtraction use the first K frames to</span>
<span style="color:#696969; "># model ad estimate the background:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; ">#  - Detection results obtained using the first K </span>
<span style="color:#696969; ">#    frames are not reliable</span>
<span style="color:#696969; ">#  - Thus, we should skip the first K frames without</span>
<span style="color:#696969; ">#    computing detection results.</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
NUM_SKIPPED_INITIAL_FRAMES <span style="color:#808030; ">=</span> <span style="color:#008c00; ">150</span>

<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Test imports and display package versions</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Testing the OpenCV version</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"OpenCV : "</span><span style="color:#808030; ">,</span>cv2<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Testing the numpy version</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Numpy : "</span><span style="color:#808030; ">,</span>np<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span>

OpenCV <span style="color:#808030; ">:</span>  <span style="color:#008000; ">3.4</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">8</span>
Numpy <span style="color:#808030; ">:</span>  <span style="color:#008000; ">1.19</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">2</span>
</pre>

### 4.2. Step 2: Read the input video:
* We now read the input video file and display its number of frames.

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#----------------------------------------------------</span>
<span style="color:#696969; "># Open and fixed-camera video file</span>
<span style="color:#696969; ">#----------------------------------------------------</span>
<span style="color:#696969; "># the source video file name</span>
video_file_path <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">"../data/OpenCV/vtest.avi"</span>
<span style="color:#696969; "># check if the reference image file exists</span>
<span style="color:#800000; font-weight:bold; ">if</span><span style="color:#808030; ">(</span>os<span style="color:#808030; ">.</span>path<span style="color:#808030; ">.</span>exists<span style="color:#808030; ">(</span>video_file_path<span style="color:#808030; ">)</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">0</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Video file name DOES NOT EXIST! = '</span> <span style="color:#44aadd; ">+</span> video_file_path<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Read the reference image</span>
cap <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>VideoCapture<span style="color:#808030; ">(</span>video_file_path<span style="color:#808030; ">)</span>
<span style="color:#696969; "># check the status of the opened video file</span>
<span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#800000; font-weight:bold; ">not</span> cap<span style="color:#808030; ">.</span>isOpened<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Cannot read video file: "</span> <span style="color:#44aadd; ">+</span> video_file_path<span style="color:#808030; ">)</span>
    exit<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
<span style="color:#696969; "># get the number of frames in the video file</span>
num_video_frames <span style="color:#808030; ">=</span> <span style="color:#400000; ">int</span><span style="color:#808030; ">(</span>cap<span style="color:#808030; ">.</span>get<span style="color:#808030; ">(</span>cv2<span style="color:#808030; ">.</span>CAP_PROP_FRAME_COUNT<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Input video file: {0} has {1} frames."</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>video_file_path<span style="color:#808030; ">,</span> num_video_frames<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>


<span style="color:#400000; ">Input</span> video <span style="color:#400000; ">file</span><span style="color:#808030; ">:</span> <span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#44aadd; ">/</span>data<span style="color:#44aadd; ">/</span>OpenCV<span style="color:#44aadd; ">/</span>vtest<span style="color:#808030; ">.</span>avi has <span style="color:#008c00; ">795</span> frames<span style="color:#808030; ">.</span>
</pre>

### 4.3. Step 3: Setup OpenCV background subtraction algorithm:

* As mentioned above, OpenCV offers 4 background-subtraction algorithms:

  * BackgroundSubtractorMOG: It is a Gaussian Mixture-based Background/Foreground Segmentation Algorithm.
  * BackgroundSubtractorMOG2: It is also a Gaussian Mixture-based Background/Foreground Segmentation Algorithm. It prov* ides better adaptability to varying scenes due illumination changes etc.
  * BackgroundSubtractorGMG: This algorithm combines statistical background image estimation and per-pixel Bayesian segmentation.
  * BackgroundSubtractorKNN: This algorithm is based on KNN clustering and classification.

Next, we instantiate the selected OpenCV background-subtraction method as defined by the OPENCV_BACKGROUND_SUBTRACTION_METHOD flag.


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------------------------</span>
<span style="color:#696969; "># Instantiate the selected type of OpenCV background-subtraction method </span>
<span style="color:#696969; "># as defined by the OPENCV_BACKGROUND_SUBTRACTION_METHOD flag.</span>
<span style="color:#696969; ">#------------------------------------------------------------------------</span>
<span style="color:#696969; "># 1) BackgroundSubtractorMOG: It is a Gaussian Mixture-based </span>
<span style="color:#696969; ">#                          Background/Foreground Segmentation Algorithm.</span>
<span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span> OPENCV_BACKGROUND_SUBTRACTION_METHOD <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">1</span> <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    fgbg <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>bgsegm<span style="color:#808030; ">.</span>createBackgroundSubtractorMOG<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span> <span style="color:#008c00; ">4</span>
    <span style="color:#696969; "># Applied BS method name</span>
    applied_bs_method_name <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">"BackgroundSubtractorMOG"</span>
<span style="color:#696969; ">#------------------------------------------------------------------------</span>
<span style="color:#696969; "># 2) BackgroundSubtractorMOG2 �" It is also a Gaussian Mixture-based </span>
<span style="color:#696969; ">#    Background/Foreground Segmentation Algorithm. It prov* ides better </span>
<span style="color:#696969; ">#    adaptability to varying scenes due illumination changes etc.</span>
<span style="color:#696969; ">#------------------------------------------------------------------------</span>
<span style="color:#800000; font-weight:bold; ">elif</span> <span style="color:#808030; ">(</span> OPENCV_BACKGROUND_SUBTRACTION_METHOD <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">2</span> <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    fgbg <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>createBackgroundSubtractorMOG2<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
    <span style="color:#696969; "># Applied BS method name</span>
    applied_bs_method_name <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">"BackgroundSubtractorMOG2"</span>
<span style="color:#696969; ">#------------------------------------------------------------------------</span>
<span style="color:#696969; "># 3) BackgroundSubtractorGMG �" This algorithm combines statistical </span>
<span style="color:#696969; ">#    background image estimation and per-pixel Bayesian segmentation.</span>
<span style="color:#696969; ">#------------------------------------------------------------------------</span>
<span style="color:#800000; font-weight:bold; ">elif</span> <span style="color:#808030; ">(</span> OPENCV_BACKGROUND_SUBTRACTION_METHOD <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">3</span> <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    fgbg <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>bgsegm<span style="color:#808030; ">.</span>createBackgroundSubtractorGMG<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
    <span style="color:#696969; "># Applied BS method name</span>
    applied_bs_method_name <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">"BackgroundSubtractorGMG"</span>
<span style="color:#696969; ">#------------------------------------------------------------------------</span>
<span style="color:#696969; "># 4) BackgroundSubtractorKNN �" This algorithm is based on KNN clustering </span>
<span style="color:#696969; ">#    and classification.</span>
<span style="color:#696969; ">#------------------------------------------------------------------------</span>
<span style="color:#800000; font-weight:bold; ">elif</span> <span style="color:#808030; ">(</span> OPENCV_BACKGROUND_SUBTRACTION_METHOD <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">4</span> <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    fgbg <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>bgsegm<span style="color:#808030; ">.</span>createBackgroundSubtractorKN<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
    <span style="color:#696969; "># Applied BS method name</span>
    applied_bs_method_name <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">"BackgroundSubtractorKNN:"</span>
<span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span>
    <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Invalid OpenCV Background-subtraction method: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>OPENCV_BACKGROUND_SUBTRACTION_METHOD<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
    <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>OPENCV_BACKGROUND_SUBTRACTION_METHOD<span style="color:#808030; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">" can only be equa to 1, 2 or 3."</span><span style="color:#808030; ">)</span>
    exit<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

### 4.3. Step 4: Apply the background subtraction on video frames

* Read the video frames one by one, sequentially
* Apply each instantiated background subtractor on each frame
* Display the background subtraction results obtained from each background subtractor for comparison.
* Post-process the detected background-subtraction results and enclose each significant detected change in:
  * Rectangular bounding-box (Red)
  * Oriented bounding-box (green)
  * Circle (Blue) : This may not be very useful so commented-out


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------</span>
<span style="color:#696969; "># Repeat for each video frame until:</span>
<span style="color:#696969; ">#------------------------------------------------</span>
<span style="color:#696969; ">#   - User presses the "ESC" key to end the </span>
<span style="color:#696969; ">#     processing</span>
<span style="color:#696969; ">#   - Or the end of video has been reached</span>
<span style="color:#696969; ">#------------------------------------------------</span>
<span style="color:#696969; "># frame counter</span>
frame_counter <span style="color:#808030; ">=</span> <span style="color:#008c00; ">0</span><span style="color:#808030; ">;</span>
<span style="color:#696969; "># start reading and processing each video frame.</span>
<span style="color:#800000; font-weight:bold; ">while</span> <span style="color:#074726; ">True</span><span style="color:#808030; ">:</span>
    <span style="color:#696969; ">#--------------------------------------------</span>
    <span style="color:#696969; "># Step 1: read the next video frame</span>
    <span style="color:#696969; ">#--------------------------------------------</span>
    ret<span style="color:#808030; ">,</span> img <span style="color:#808030; ">=</span> cap<span style="color:#808030; ">.</span>read<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
      
    <span style="color:#696969; ">#--------------------------------------------</span>
    <span style="color:#696969; "># Step 2: Apply mask for background </span>
    <span style="color:#696969; ">#         subtraction</span>
    <span style="color:#696969; ">#--------------------------------------------</span>
    <span style="color:#696969; "># Apply the selected background-subtraction </span>
    <span style="color:#696969; "># algorithm</span>
    <span style="color:#696969; ">#--------------------------------------------</span>
    fgmask <span style="color:#808030; ">=</span> fgbg<span style="color:#808030; ">.</span>apply<span style="color:#808030; ">(</span>img<span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
      
    <span style="color:#696969; ">#------------------------------------------------------</span>
    <span style="color:#696969; "># Background subtraction use the first K frames to</span>
    <span style="color:#696969; "># model ad estimate the background:</span>
    <span style="color:#696969; ">#------------------------------------------------------</span>
    <span style="color:#696969; ">#  - Detection results obtained using the first K </span>
    <span style="color:#696969; ">#    frames are not reliable</span>
    <span style="color:#696969; ">#  - Thus, we should skip the first K frames without</span>
    <span style="color:#696969; ">#    computing detection results</span>
    <span style="color:#696969; ">#------------------------------------------------------</span>
    <span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span> frame_counter <span style="color:#44aadd; ">&gt;</span> NUM_SKIPPED_INITIAL_FRAMES <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># Step 4) Enclose each detected change in:</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># - Rectangular bounding-box (Red)</span>
        <span style="color:#696969; "># - Oriented bounding-box (green)</span>
        <span style="color:#696969; "># - Circle (Blue)</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># 4.1) First, find contours and get the </span>
        <span style="color:#696969; ">#      external one</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        ret1<span style="color:#808030; ">,</span> contours<span style="color:#808030; ">,</span> ret3 <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>findContours<span style="color:#808030; ">(</span>fgmask<span style="color:#808030; ">,</span> cv2<span style="color:#808030; ">.</span>RETR_TREE<span style="color:#808030; ">,</span> cv2<span style="color:#808030; ">.</span>CHAIN_APPROX_SIMPLE<span style="color:#808030; ">)</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># 4.2) Iterate over the contours and draw the </span>
        <span style="color:#696969; ">#      enclosing shapes, as mentioned above.</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#800000; font-weight:bold; ">for</span> c <span style="color:#800000; font-weight:bold; ">in</span> contours<span style="color:#808030; ">:</span>
            <span style="color:#696969; ">#----------------------------------------</span>
            <span style="color:#696969; "># 4.2.1) Get the are of the contour</span>
            <span style="color:#696969; ">#----------------------------------------</span>
            contour_area <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>contourArea<span style="color:#808030; ">(</span>c<span style="color:#808030; ">)</span>
            <span style="color:#696969; ">#----------------------------------------</span>
            <span style="color:#696969; "># Only draw oncours with significant areas</span>
            <span style="color:#696969; ">#----------------------------------------</span>
            <span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span> contour_area <span style="color:#44aadd; ">&gt;=</span> MIN_CONTOUR_AREA <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
                <span style="color:#696969; ">#----------------------------------------</span>
                <span style="color:#696969; "># 4.2.1.1) Display the contour on the </span>
                <span style="color:#696969; ">#         frame image in YELLO</span>
                <span style="color:#696969; ">#----------------------------------------</span>
                <span style="color:#696969; "># cv2.drawContours(img, c, -1, (255, 255, 0), 1)</span>

                <span style="color:#696969; ">#----------------------------------------</span>
                <span style="color:#696969; "># 4.2.1.2) Get the rectangular bounding </span>
                <span style="color:#696969; ">#        boxes: cv2.boundingRect</span>
                <span style="color:#696969; ">#----------------------------------------</span>
                x<span style="color:#808030; ">,</span> y<span style="color:#808030; ">,</span> w<span style="color:#808030; ">,</span> h <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>boundingRect<span style="color:#808030; ">(</span>c<span style="color:#808030; ">)</span>
                <span style="color:#696969; "># draw a RED rectangle to visualize the bounding rect</span>
                cv2<span style="color:#808030; ">.</span>rectangle<span style="color:#808030; ">(</span>img<span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span>x<span style="color:#808030; ">,</span> y<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span>x<span style="color:#44aadd; ">+</span>w<span style="color:#808030; ">,</span> y<span style="color:#44aadd; ">+</span>h<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">255</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span>

                <span style="color:#696969; ">#----------------------------------------</span>
                <span style="color:#696969; "># 4.2.1.3) Get the min-area oriented bounding </span>
                <span style="color:#696969; ">#        boxes: cv2.minAreaRect()</span>
                <span style="color:#696969; ">#----------------------------------------</span>
                <span style="color:#696969; "># get the min area rect</span>
                rect <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>minAreaRect<span style="color:#808030; ">(</span>c<span style="color:#808030; ">)</span>
                box <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>boxPoints<span style="color:#808030; ">(</span>rect<span style="color:#808030; ">)</span>
                <span style="color:#696969; "># convert all coordinates floating point values to int</span>
                box <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>int0<span style="color:#808030; ">(</span>box<span style="color:#808030; ">)</span>
                <span style="color:#696969; "># draw a GREEN 'oreinted-rectangle</span>
                cv2<span style="color:#808030; ">.</span>drawContours<span style="color:#808030; ">(</span>img<span style="color:#808030; ">,</span> <span style="color:#808030; ">[</span>box<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">255</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span>

                <span style="color:#696969; ">#----------------------------------------</span>
                <span style="color:#696969; "># 4.1.1.3) Get the min-circle: </span>
                <span style="color:#696969; ">#        cv2.minEnclosingCircle()</span>
                <span style="color:#696969; ">#----------------------------------------</span>
                <span style="color:#696969; "># this is not very useful!</span>
                <span style="color:#696969; ">#----------------------------------------</span>
                <span style="color:#696969; "># finally, get the min enclosing circle</span>
                <span style="color:#696969; "># (x, y), radius = cv2.minEnclosingCircle(c)</span>
                <span style="color:#696969; "># convert all values to int</span>
                <span style="color:#696969; "># center = (int(x), int(y))</span>
                <span style="color:#696969; "># radius = int(radius)</span>
                <span style="color:#696969; "># and draw the circle in BLUE</span>
                <span style="color:#696969; "># img = cv2.circle(img, center, radius, (255, 0, 0), 2)</span>
                <span style="color:#696969; ">#----------------------------------------</span>

        <span style="color:#696969; ">#----------------------------------------</span>
        <span style="color:#696969; "># display the frame image with the </span>
        <span style="color:#696969; "># overlaid contours</span>
        <span style="color:#696969; ">#----------------------------------------</span>
        <span style="color:#696969; "># cv2.drawContours(img, contours, -1, (255, 255, 0), 1)</span>
        cv2<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Background-Subtraction Method: "</span> <span style="color:#44aadd; ">+</span> applied_bs_method_name<span style="color:#808030; ">,</span> img<span style="color:#808030; ">)</span>

        <span style="color:#696969; "># save the frame with overlay for multiple of 100</span>
        <span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span> frame_counter <span style="color:#44aadd; ">&gt;</span> <span style="color:#008c00; ">0</span> <span style="color:#800000; font-weight:bold; ">and</span> frame_counter <span style="color:#44aadd; ">%</span> <span style="color:#008c00; ">100</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">0</span> <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
            <span style="color:#696969; "># the source video file name</span>
            output_file_path <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">"../results/OpenCV/frame-"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>frame_counter<span style="color:#808030; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">".jpg"</span>
            <span style="color:#696969; "># save the frame</span>
            cv2<span style="color:#808030; ">.</span>imwrite<span style="color:#808030; ">(</span>output_file_path<span style="color:#808030; ">,</span> img<span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>

    <span style="color:#696969; "># increment the frame counter</span>
    frame_counter <span style="color:#808030; ">=</span> frame_counter <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">;</span>
    
    <span style="color:#696969; ">#----------------------------------------</span>
    <span style="color:#696969; "># check if the total number of video frames </span>
    <span style="color:#696969; "># has been reached:</span>
    <span style="color:#696969; "># - if so, stop processing!</span>
    <span style="color:#696969; ">#----------------------------------------</span>
    <span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span> frame_counter <span style="color:#44aadd; ">&gt;=</span> num_video_frames <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
        <span style="color:#696969; "># stop processing</span>
        <span style="color:#800000; font-weight:bold; ">break</span><span style="color:#808030; ">;</span>
    
    <span style="color:#696969; "># press ESC to terminate the processing</span>
    k <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>waitKey<span style="color:#808030; ">(</span><span style="color:#008c00; ">30</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">&amp;</span> <span style="color:#008c00; ">0xff</span><span style="color:#808030; ">;</span>
    <span style="color:#800000; font-weight:bold; ">if</span> k <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">27</span><span style="color:#808030; ">:</span>
        <span style="color:#800000; font-weight:bold; ">break</span><span style="color:#808030; ">;</span>

<span style="color:#696969; "># clear the video capture object</span>
cap<span style="color:#808030; ">.</span>release<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
<span style="color:#696969; "># close all windows</span>
cv2<span style="color:#808030; ">.</span>destroyAllWindows<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

### 4.4. Step 4: Visualize the bounding-boxes of the detected templates:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#----------------------------------------------------</span>
<span style="color:#696969; "># Process the detected templates and overlay the </span>
<span style="color:#696969; "># their bounding-boxes on the source image</span>
<span style="color:#696969; ">#----------------------------------------------------</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'-----------------------------------------------------------------------'</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Template matching resulst with: CORRELATION-COEFFICENT THRESHOLD= {0}'</span><span style="color:#808030; ">.\\
</span>format<span style="color:#808030; ">(</span>CCOEFF_NORMED_THRESHOLD<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'-----------------------------------------------------------------------'</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#----------------------------------------------------</span>
<span style="color:#696969; "># - First get the location detected with the</span>
<span style="color:#696969; ">#   similarity metric</span>
<span style="color:#696969; ">#----------------------------------------------------</span>
min_val<span style="color:#808030; ">,</span> max_val<span style="color:#808030; ">,</span> min_loc<span style="color:#808030; ">,</span> max_loc <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>minMaxLoc<span style="color:#808030; ">(</span>det_results<span style="color:#808030; ">)</span>
<span style="color:#696969; "># get the TLC of the highest-confidence bounding-box</span>
top_left <span style="color:#808030; ">=</span> max_loc
<span style="color:#696969; "># get the BRC of the highest-confidence bounding-box</span>
bottom_right <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span>top_left<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">+</span> template_img_height<span style="color:#808030; ">,</span> top_left<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">+</span> template_img_width<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Check if the cross-correlation value is greater than the </span>
<span style="color:#696969; "># specified cross-correlation threshold, then plot it on the image</span>
<span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span> max_val <span style="color:#44aadd; ">&gt;=</span> CCOEFF_NORMED_THRESHOLD <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    <span style="color:#696969; "># overlay the bounding-box in: GREEN color</span>
    cv2<span style="color:#808030; ">.</span>rectangle<span style="color:#808030; ">(</span>reference_img<span style="color:#808030; ">,</span>top_left<span style="color:#808030; ">,</span> bottom_right<span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># display a message</span>
    <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'BEST Match: Template is detected source image with MAX-CORRELATION-COEFFICENT = {0}'</span><span style="color:#808030; ">.\\
</span>format<span style="color:#808030; ">(</span>max_val<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#----------------------------------------------------</span>
<span style="color:#696969; "># - Now check if there are any other template</span>
<span style="color:#696969; ">#   detection results</span>
<span style="color:#696969; ">#---------------------------------------------------- </span>
<span style="color:#696969; "># find all the detection with cross-correlation execcding the</span>
<span style="color:#696969; "># specified cross-correlation threshold, then plot it on the image</span>
x_locs<span style="color:#808030; ">,</span> y_locs <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>where<span style="color:#808030; ">(</span> det_results <span style="color:#44aadd; ">&gt;=</span> CCOEFF_NORMED_THRESHOLD<span style="color:#808030; ">)</span>
<span style="color:#696969; "># the number of detected templates</span>
num_detected_templates <span style="color:#808030; ">=</span> x_locs<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"The total number of detected templates = {0}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>num_detected_templates<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">for</span> counter <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_detected_templates<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    <span style="color:#696969; "># Ignore the best-detection, which is already plotted</span>
    <span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span> y_locs<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span> <span style="color:#44aadd; ">!=</span> max_loc<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> <span style="color:#800000; font-weight:bold; ">and</span> x_locs<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span> <span style="color:#44aadd; ">!=</span> max_loc<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
        <span style="color:#696969; "># the TLC of the bbox</span>
        top_left <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span>y_locs<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> x_locs<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; "># the BRC of the bbox</span>
        bottom_right <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span>top_left<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">+</span> template_img_height<span style="color:#808030; ">,</span> top_left<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">+</span> template_img_width<span style="color:#808030; ">)</span>
        <span style="color:#696969; "># overlay the bbox</span>
        cv2<span style="color:#808030; ">.</span>rectangle<span style="color:#808030; ">(</span>reference_img<span style="color:#808030; ">,</span>top_left<span style="color:#808030; ">,</span> bottom_right<span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; "># display a message</span>
        <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'OTHER Match: Template is detected source image with MAX-CORRELATION-COEFFICENT = {0}'</span><span style="color:#808030; ">.</span>\\
format<span style="color:#808030; ">(</span>max_val<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
        
<span style="color:#696969; ">#---------------------------------------------------- </span>
<span style="color:#696969; "># - Visualize the final template detection results</span>
<span style="color:#696969; ">#---------------------------------------------------- </span>
<span style="color:#696969; "># create the figure</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Template Matching Results"</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
<span style="color:#696969; "># display the figure</span>
plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>cv2<span style="color:#808030; ">.</span>cvtColor<span style="color:#808030; ">(</span>reference_img<span style="color:#808030; ">,</span> cv2<span style="color:#808030; ">.</span>COLOR_BGR2RGB<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
<span style="color:#696969; "># display the figure</span>
plt<span style="color:#808030; ">.</span>show<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
<span style="color:#696969; "># axis off</span>
plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>


<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Template matching resulst <span style="color:#800000; font-weight:bold; ">with</span><span style="color:#808030; ">:</span> CORRELATION<span style="color:#44aadd; ">-</span>COEFFICENT THRESHOLD<span style="color:#808030; ">=</span> <span style="color:#008000; ">0.99</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
BEST Match<span style="color:#808030; ">:</span> Template <span style="color:#800000; font-weight:bold; ">is</span> detected source image <span style="color:#800000; font-weight:bold; ">with</span> <span style="color:#400000; ">MAX</span><span style="color:#44aadd; ">-</span>CORRELATION<span style="color:#44aadd; ">-</span>COEFFICENT <span style="color:#808030; ">=</span> <span style="color:#008000; ">0.9956235885620117</span>
The total number of detected templates <span style="color:#808030; ">=</span> <span style="color:#008c00; ">1</span>
</pre>

### 4.5. Step 5: Display a successful execution message:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># display a final message</span>
<span style="color:#696969; "># current time</span>
now <span style="color:#808030; ">=</span> datetime<span style="color:#808030; ">.</span>datetime<span style="color:#808030; ">.</span>now<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>now<span style="color:#808030; ">.</span>strftime<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">"</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Program executed successfully on<span style="color:#808030; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">03</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">30</span> <span style="color:#008c00; ">07</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">54</span><span style="color:#808030; ">:</span><span style="color:#008000; ">06.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>Goodbye!
</pre>


## 5. Sample Detection Results

* Sample detection results using OpenCV BackgroundSubtractorMOG() background-subtractor are illustrated below:

  * The detection results are post-processed and localized using:
    * Rectangular bounding-boxes (red)
    * Oriented minimum-area bounding boxes (green).

  * All people in the scene have been correctly detected:
    * Generally, the bounding boxes fit well to the full body of the person
    * In a few cases, only part if a person's body is enclosed within a bounding box.
    * In some cases, when a few people are too close together, they are merged together into a single bonding-box.
    * Generally, the detection results are very good, using the simplest OpenCV background subtractor (BackgroundSubtractorMOG)
    * Better detection results were observed using the other 3 more complex and accurate background subtraction algorithms.

<img src="images/sample-detection-results.PNG" width="1000"/>

## 6. Analysis

We have demonstrated that template matching works well for detecting a fixed-template from a course, acquired under similar conditions. 
However, template matching has several limitations, including:

  * Sensitive to changes in illumination, scale and orientation.
  * Sensitive to occlusion, as the object needs to be fully visible in the scene image to be detected.
  * Pattern occurrences have to preserve the orientation of the reference pattern image.  As a result, it does not work for rotated or scaled versions of the template as a change in shape/size/shear etc. of object with respect to template will give a false match.
  * The method is inefficient when calculating the pattern correlation image for medium to large images as the process is time consuming. 
  * Template matching  techniques are more suitable for restricted environments where imaging conditions, such as image intensity and viewing angles between the template and images containing this template are the same. 
  * However, recently template-matching techniques, which are less sensitive to variations in scale, translation, brightness and contrast have been proposed with some reported success. 
  * For example, multi-scale template matching is able of detect templates at different scales. That is if the size of the source image patch corresponding to the template image has different size and scale than the used template image.


## 7. Future Work

* We propose to explore the following related issues:

  * Use annotated data set to evaluate and compare the performance of the various OpenCV background-subtractor to detect changes and moving objects in the video scene:
  * Generate quantitative performance metrics for each algorithm, such as ROC and PR curves, and compare the results
  * Also compare the computation complexity of each algorithm
  * Experiment with changing some of the configuration parameters, for each algorithm, when have been set to their default values so far.


## 8. References

1. OpenCV. How to Use Background Subtraction Methods . https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html
2. OpenCV. Background Subtraction. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html
3. OpenCV. How to Use Background Subtraction Methods. https://www.ccoderun.ca/programming/doxygen/opencv/tutorial_background_subtraction.html 
4. GeeksforGeeks. Background subtraction - OpenCV. https://www.geeksforgeeks.org/background-subtraction-opencv/
5. GeekforGeeks. Python | Background subtraction using OpenCV. https://www.geeksforgeeks.org/python-background-subtraction-using-opencv/ 
6. Anastasia Murzova. Background Subtraction with OpenCV and BGS Libraries. https://learnopencv.com/background-subtraction-with-opencv-and-bgs-libraries/

