from PIL import Image
import numpy as np
from skimage import feature
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def canny_edge_detection(image_path, sigma=1.0):
    # Load and convert image to grayscale
    image = Image.open(image_path).convert("L")
    image_np = np.array(image)
    
    # Apply Gaussian filter for smoothing
    smoothed_image_np = gaussian_filter(image_np, sigma=sigma)
    
    # Perform Canny edge detection
    edges = feature.canny(smoothed_image_np)
    
    # Convert edges to image format
    edges_image_np = np.uint8(edges * 255)
    edges_image = Image.fromarray(edges_image_np, mode="L")
    
    return edges_image

def display_image(image):
    # Display the image using matplotlib
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.show()

# Example usage
image_path = 'path/to/your/image.jpg'  
sigma = 1.0  # Parameter

edges_image = canny_edge_detection(image_path, sigma)
display_image(edges_image)
