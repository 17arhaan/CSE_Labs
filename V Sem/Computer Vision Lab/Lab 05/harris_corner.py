import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris_corner_detection(image_path, k=0.04, threshold=0.01):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Compute image gradients
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute products of gradients
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    
    # Apply Gaussian filter
    Sx2 = cv2.GaussianBlur(Ix2, (3, 3), 0)
    Sy2 = cv2.GaussianBlur(Iy2, (3, 3), 0)
    Sxy = cv2.GaussianBlur(Ixy, (3, 3), 0)
    
    # Compute Harris response
    det = (Sx2 * Sy2) - (Sxy * Sxy)
    trace = Sx2 + Sy2
    R = det - k * (trace * trace)
    
    # Thresholding
    R_thresholded = np.copy(R)
    R_thresholded[R < threshold * R.max()] = 0
    
    # Draw corners
    corners = np.argwhere(R_thresholded > 0)
    
    # Convert image back to color for visualization
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for corner in corners:
        y, x = corner
        cv2.circle(color_image, (x, y), 5, (0, 255, 0), 1)
    
    # Display the result
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    plt.title('Harris Corners')
    plt.axis('off')
    plt.show()

harris_corner_detection('/home/student/Desktop/220962050/Computer Vision Lab/Lab 05/reference.jpg')
