import cv2
image = cv2.imread('img.png')
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
theta = 90
scale = 1.0
rotation = cv2.getRotationMatrix2D(center, theta, scale)
cos = abs(rotation[0, 0])
sin = abs(rotation[0, 1])
new_w = int(h * sin + w * cos)
new_h = int(h * cos + w * sin)
rotation[0, 2] += (new_w / 2) - center[0]
rotation[1, 2] += (new_h / 2) - center[1]
rotated_image = cv2.warpAffine(image, rotation, (new_w, new_h))
cv2.imwrite('new_rotate.png', rotated_image)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()