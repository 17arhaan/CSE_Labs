import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate or load your data
# For simplicity, we'll create synthetic 2D data
np.random.seed(0)
data = np.vstack([
    np.random.normal(loc=[0, 0], scale=1, size=(100, 2)),
    np.random.normal(loc=[5, 5], scale=1, size=(100, 2)),
    np.random.normal(loc=[-5, 5], scale=1, size=(100, 2))
])

# Step 2: Prepare the data
# K-means expects a 2D array where each row is a data point
data = np.float32(data)

# Step 3: Apply K-means clustering
K = 3  # Number of clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
flags = cv2.KMEANS_RANDOM_CENTERS

compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 10, flags)

# Step 4: Visualize the clusters
plt.scatter(data[:, 0], data[:, 1], c=labels.flatten(), cmap='viridis', marker='o')
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='x', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
