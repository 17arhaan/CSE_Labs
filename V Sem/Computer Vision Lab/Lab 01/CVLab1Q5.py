import cv2
image = cv2.imread('img.png') # OPen img
original_height, original_width = image.shape[:2] # Extract height and width
new_width = int(original_width * 1.25) # Increase 1.25 times
new_height = int(original_height * 1.25) # Increase 1.25 times
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
cv2.imwrite('new_resize.png', resized_image)
cv2.imshow('Increased Size Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()