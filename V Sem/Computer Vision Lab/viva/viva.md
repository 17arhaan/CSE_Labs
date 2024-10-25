
# Lab 5: Implementation of Feature Extraction Methods

### 1. **Overview of Feature Extraction**
   - **Purpose:** Extract essential information from images that is more informative and non-redundant, to facilitate subsequent learning and generalization steps.

### 2. **Key Techniques Explained**
   - **Histogram of Oriented Gradients (HOG):**
     - **Procedure:** Divide the image into small regions (cells), for each cell compute a histogram of gradient directions or edge orientations using `cv2.calcHist`.
     - **Application:** Widely used for object detection, particularly for pedestrian detection in computer vision.
   - **Scale-Invariant Feature Transform (SIFT):**
     - **Steps:**
       - Detect potential interest points using `cv2.SIFT_create`.
       - Localize keypoints and generate descriptors with `cv2.SIFT.detectAndCompute`.
     - **Application:** Feature matching across different images, object recognition, 3D reconstruction.
   - **Local Binary Pattern (LBP):** Not directly supported in OpenCV, but can be implemented using `cv2.threshold` and custom routines.

### 3. **Practical Exercises**
   - **Implementing and Visualizing HOG for Object Detection**
   - **Developing SIFT-based Feature Matching System**
   - **Using LBP for Robust Texture Classification**

# Lab 6: Implementation of Feature Matching Methods

### 1. **Introduction to Feature Matching**
   - **Purpose:** Identify and match individual features between different images based on their descriptors, essential for motion tracking, image stitching, and stereo vision.

### 2. **Advanced Matching Techniques**
   - **Brute-Force Matcher:**
     - **Description:** Uses `cv2.BFMatcher` to compare each descriptor in the first set with all descriptors in the second set and finds the closest one.
     - **Use Case:** Best for small datasets where precision is more critical than speed.
   - **FLANN Matcher:**
     - **Description:** Utilizes `cv2.FlannBasedMatcher` with optimized algorithms to find good matches quickly.
     - **Use Case:** Feature matching in real-time applications.
   - **RANSAC:**
     - **Process:** Often used with `cv2.findHomography` to robustly estimate a homography matrix that aligns matched features.
     - **Use Case:** Robust estimation problems such as camera calibration and 3D reconstruction.

### 3. **Lab Exercises**
   - **Implementing and Comparing Various Matching Algorithms**
   - **Using RANSAC for Robust Estimation of Geometric Transformations**
   - **Performance Evaluation of Feature Matching under Different Conditions**

# Lab 7: Implementation of Camera Calibration

### 1. **Concept of Camera Calibration**
   - **Purpose:** Determine the camera's intrinsic (focal length, principal point, skew) and extrinsic (orientation and position) parameters, essential for accurate 3D scene interpretation from 2D images.

### 2. **Techniques for Calibration**
   - **Pinhole Camera Model:**
     - **Description:** Assumes a simple geometric model where light passes through a single point (pinhole).
   - **Calibration Process:**
     - **Procedure:** Use multiple images of a known calibration pattern (checkerboard), and use `cv2.calibrateCamera` to compute the camera parameters.
     - **Output:** Optimized camera matrix, distortion coefficients, rotation and translation vectors using `cv2.calibrateCamera`.

### 3. **Exercises and Applications**
   - **Estimating and Validating Camera Parameters**
   - **Assessing the Accuracy of Calibration through Reprojection Errors**
   - **Applications in Robotics, Augmented Reality, and Photogrammetry**
