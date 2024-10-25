# Computer Vision Lab Summary

## Lab 5: Implementation of Feature Extraction Methods

### 1. **Overview of Feature Extraction**
   - **Purpose:** Extract essential information from images that is more informative and non-redundant, to facilitate subsequent learning and generalization steps.

### 2. **Key Techniques Explained**
   - **Histogram of Oriented Gradients (HOG):**
     - **Procedure:** Divide the image into small regions (cells), for each cell compute a histogram of gradient directions or edge orientations.
     - **Application:** Widely used for object detection, particularly for pedestrian detection in computer vision.
   - **Scale-Invariant Feature Transform (SIFT):**
     - **Steps:**
       - Detect potential interest points that are invariant to scale and orientation.
       - Localize keypoints to sub-pixel accuracy and filter out low contrast points.
       - Assign orientation based on local image gradient.
       - Generate a descriptor that is robust to local geometric and photometric transformations.
     - **Application:** Feature matching across different images, object recognition, 3D reconstruction.
   - **Local Binary Pattern (LBP):**
     - **Procedure:** For each pixel in the image, compare it with its neighbors and assign a binary code.
     - **Application:** Texture classification, face recognition, and other applications where illumination variations are a concern.

### 3. **Practical Exercises**
   - **Implementing and Visualizing HOG for Object Detection**
   - **Developing SIFT-based Feature Matching System**
   - **Using LBP for Robust Texture Classification**

## Lab 6: Implementation of Feature Matching Methods

### 1. **Introduction to Feature Matching**
   - **Purpose:** Identify and match individual features between different images based on their descriptors, essential for motion tracking, image stitching, and stereo vision.

### 2. **Advanced Matching Techniques**
   - **Brute-Force Matcher:**
     - **Description:** Compares each descriptor in the first set with all descriptors in the second set and finds the closest one.
     - **Use Case:** Best for small datasets where precision is more critical than speed.
   - **FLANN Matcher:**
     - **Description:** Utilizes optimized algorithms to find good matches quickly, suitable for large datasets.
     - **Use Case:** Feature matching in real-time applications.
   - **RANSAC:**
     - **Process:** Randomly select data points to estimate a model and classify all points based on their fit to the model, iterating this process to maximize the number of inliers.
     - **Use Case:** Robust estimation problems such as camera calibration and 3D reconstruction.

### 3. **Lab Exercises**
   - **Implementing and Comparing Various Matching Algorithms**
   - **Using RANSAC for Robust Estimation of Geometric Transformations**
   - **Performance Evaluation of Feature Matching under Different Conditions**

## Lab 7: Implementation of Camera Calibration

### 1. **Concept of Camera Calibration**
   - **Purpose:** Determine the camera's intrinsic (focal length, principal point, skew) and extrinsic (orientation and position) parameters, essential for accurate 3D scene interpretation from 2D images.

### 2. **Techniques for Calibration**
   - **Pinhole Camera Model:**
     - **Description:** Assumes a simple geometric model where light passes through a single point (pinhole).
     - **Role:** Foundation for understanding more complex camera models and for initial estimations in calibration algorithms.
   - **Calibration Process:**
     - **Procedure:** Use multiple images of a known calibration pattern (checkerboard), compute the homography, and solve for the camera parameters.
     - **Output:** Optimized camera matrix, distortion coefficients, rotation and translation vectors.

### 3. **Exercises and Applications**
   - **Estimating and Validating Camera Parameters**
   - **Assessing the Accuracy of Calibration through Reprojection Errors**
   - **Applications in Robotics, Augmented Reality, and Photogrammetry**
