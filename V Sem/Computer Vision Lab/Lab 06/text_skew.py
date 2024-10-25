import cv2
import numpy as np
from scipy import stats

def hough_transform(image, threshold):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

    img_lines = image.copy()
    slopes = []  

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if x2 != x1: 
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = float('inf') 

            slopes.append(slope)

    angles = [np.degrees(np.arctan(slope)) if slope != float('inf') else 90 for slope in slopes]

    angles = [angle if -90 <= angle <= 90 else (angle - 180 if angle > 90 else angle + 180) for angle in angles]

    mode_slope = None
    if slopes:
        mode_result = stats.mode(slopes)
        mode_slope = mode_result.mode  

    return img_lines, slopes, angles, mode_slope


def rotate_image(image, angle):
    (h, w) = image.shape[:2]

    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    return rotated

img = cv2.imread('img.png')  

new_img, slopes, angles, mode_slope = hough_transform(img, 100)

print('Slopes of detected lines:', slopes)
print('Angles of detected lines (degrees):', angles)
print('Mode of slopes:', mode_slope)
mode_angle = np.degrees(np.arctan(mode_slope))
print(mode_angle)

rotated_img = rotate_image(img, mode_angle)

cv2.imshow('Detected Lines', rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()