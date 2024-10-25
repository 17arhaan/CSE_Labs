import cv2
import numpy as np

# Load two consecutive frames (you can use any video or image sequence)
# Here, we are using a video file; you can replace this with your images
cap = cv2.VideoCapture('car.mp4')  # Replace with your video path

# Read the first frame
ret, old_frame = cap.read()
if not ret:
    print("Error: Cannot read video.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Convert the frame to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Create a mask for drawing purposes
mask = np.zeros_like(old_frame)

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Define the feature parameters to detect good features to track
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Detect good features to track
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the new frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None and p0 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)  # Ensure points are integers
            c, d = old.ravel().astype(int)  # Ensure points are integers
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    # Overlay the mask on the current frame
    img = cv2.add(frame, mask)

    # Display the result
    cv2.imshow('Optical Flow - Lucas-Kanade', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Release resources and close all windows
cap.release()
cv2.destroyAllWindows()