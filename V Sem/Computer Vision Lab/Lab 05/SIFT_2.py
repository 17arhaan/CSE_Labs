import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def match_images(image1_path, image2_path):
    """Perform SIFT feature matching on two images."""
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    keypoints1, descriptors1 = detect_and_compute_sift(image1)
    keypoints2, descriptors2 = detect_and_compute_sift(image2)

    matches = match_descriptors(descriptors1, descriptors2)
    good_matches = filter_matches(matches)

    img_matches = draw_matches(image1, keypoints1, image2, keypoints2, good_matches)
    
    # Display the matched image
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title('Feature Matching')
    plt.show()

# Example usage:
image1_path = "path/to/first/image.jpg"
image2_path = "path/to/second/image.jpg"
match_images(image1_path, image2_path)