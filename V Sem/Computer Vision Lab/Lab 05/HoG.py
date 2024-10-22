import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import glob

def compute_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2)).flatten()

def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def calculate_similarity(hog1, hog2):
    return cosine_similarity([hog1], [hog2])[0][0]

def non_max_suppression(detections, overlap_threshold=0.3):
    if not detections: return []
    boxes = np.array([[x, y, x + w, y + h] for (x, y, (w, h), _) in detections])
    scores = np.array([score for (_, _, _, score) in detections])
    idxs = np.argsort(scores)[::-1]
    picked = []

    while idxs.size > 0:
        last = len(idxs) - 1
        i = idxs[last]
        picked.append(i)
        xx1 = np.maximum(boxes[i][0], boxes[idxs[:last + 1], 0])
        yy1 = np.maximum(boxes[i][1], boxes[idxs[:last + 1], 1])
        xx2 = np.minimum(boxes[i][2], boxes[idxs[:last + 1], 2])
        yy2 = np.minimum(boxes[i][3], boxes[idxs[:last + 1], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / ((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    return detections[picked]

reference_images = glob.glob('reference1.jpg')
reference_hogs = [compute_hog(cv2.imread(img)) for img in reference_images if cv2.imread(img) is not None]

if not reference_hogs:
    print("Error: No valid reference images found.")
    exit()

test_image = cv2.imread('reference.jpg')
if test_image is None:
    print("Error: Could not read the image.")
    exit()

window_size = (64, 128)
step_size = 16
threshold = 0.5
detected_windows = []

for (x, y, window) in sliding_window(test_image, step_size, window_size):
    window_hog = compute_hog(window)
    for reference_hog in reference_hogs:
        if calculate_similarity(window_hog, reference_hog) > threshold:
            detected_windows.append((x, y, window_size, _))

best_windows = non_max_suppression(detected_windows)

for (x, y, (w, h), _) in best_windows:
    cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
