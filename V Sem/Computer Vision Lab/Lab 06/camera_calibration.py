import cv2
import numpy as np
import glob

# Define the chessboard size (number of inner corners per row and column)
chessboard_size = (9, 6)  # 9x6 is commonly used, adjust for your pattern
square_size = 1.0  # Size of each square on the chessboard (in any consistent unit)

# Prepare object points (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in the real world
imgpoints = []  # 2D points in the image plane

# Load images from a folder
images = glob.glob('Skew.png')  # Adjust path to your folder containing calibration images

# Iterate over each image for calibration
for fname in images:
    image = cv2.imread(fname)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)

    # If found, add object points and image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
        cv2.imshow('Chessboard Corners', image)
        cv2.waitKey(500)  # Show each image for 500 ms
cv2.waitKey(0)
cv2.destroyAllWindows()