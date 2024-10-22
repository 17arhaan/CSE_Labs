import cv2
image = cv2.imread('img.png') # Open
b, g, r = cv2.split(image) # Split Func for giving streams of data
print(f"{b} : Blue")
print(f"{r} : Red")
print(f"{g} : Green")
cv2.imshow('Image', image) # Image View
cv2.waitKey(0) # Static Image
cv2.destroyAllWindows() # Close All