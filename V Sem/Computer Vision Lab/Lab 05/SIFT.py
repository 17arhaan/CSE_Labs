import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from io import BytesIO

def load_image(image_path):
    """Load an image from a file."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    return image

def detect_and_compute_sift(image):
    """Detect SIFT keypoints and compute descriptors."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_descriptors(descriptors1, descriptors2):
    """Match descriptors using FLANN-based matcher."""
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    return matches

def filter_matches(matches, ratio_thresh=0.75):
    """Apply the ratio test to filter good matches."""
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    return good_matches

def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    """Draw matches between two images."""
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches

def display_image(image):
    """Convert an OpenCV image to a format suitable for tkinter and display it."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    tk_image = ImageTk.PhotoImage(pil_image)
    return tk_image

def on_match_images():
    """Handle the SIFT matching process when the button is pressed."""
    global img1_path, img2_path, result_label
    
    if not img1_path or not img2_path:
        result_label.config(text="Please select both images.")
        return
    
    image1 = load_image(img1_path)
    image2 = load_image(img2_path)

    keypoints1, descriptors1 = detect_and_compute_sift(image1)
    keypoints2, descriptors2 = detect_and_compute_sift(image2)

    matches = match_descriptors(descriptors1, descriptors2)
    good_matches = filter_matches(matches)

    img_matches = draw_matches(image1, keypoints1, image2, keypoints2, good_matches)
    img_matches_tk = display_image(img_matches)

    result_label.config(image=img_matches_tk)
    result_label.image = img_matches_tk
    result_label.config(text="")

def select_image1():
    """Select the first image and update the path."""
    global img1_path
    img1_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if img1_path:
        image1_label.config(text=f"Selected Image 1: {img1_path}")

def select_image2():
    """Select the second image and update the path."""
    global img2_path
    img2_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if img2_path:
        image2_label.config(text=f"Selected Image 2: {img2_path}")

# Initialize the Tkinter window
window = tk.Tk()
window.title("SIFT Feature Matching")
window.geometry('1200x800')

# Initialize global variables
img1_path = ""
img2_path = ""

# Create and place widgets
tk.Button(window, text="Select Image 1", command=select_image1).pack(pady=10)
image1_label = tk.Label(window, text="No image selected")
image1_label.pack(pady=5)

tk.Button(window, text="Select Image 2", command=select_image2).pack(pady=10)
image2_label = tk.Label(window, text="No image selected")
image2_label.pack(pady=5)

tk.Button(window, text="Match Images", command=on_match_images).pack(pady=20)

result_label = tk.Label(window)
result_label.pack(pady=20)

window.mainloop()
