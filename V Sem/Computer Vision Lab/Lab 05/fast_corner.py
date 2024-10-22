import cv2
import numpy as np
import matplotlib.pyplot as plt

def fast_corner_detection(image_path, threshold=30):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Initialize FAST detector
    fast = cv2.FastFeatureDetector_create(threshold)

    # Detect keypoints
    keypoints = fast.detect(image, None)
    
    # Draw keypoints
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    color_image = cv2.drawKeypoints(color_image, keypoints, None, color=(0, 255, 0))
    
    # Display the result
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    plt.title('FAST Keypoints')
    plt.axis('off')
    plt.show()

fast_corner_detection('/home/student/Desktop/220962050/Computer Vision Lab/Lab 05/reference.jpg')
