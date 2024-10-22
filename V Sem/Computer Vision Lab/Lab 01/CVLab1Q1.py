import cv2
image = cv2.imread('img.png') # Read
cv2.imshow('Photo', image) # New Windoe for image
new_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to greyscale
cv2.imwrite('/home/student/220962046_Amulya_CVLab/pythonProject/new.png', new_img) # Write the new imagee
cv2.waitKey(0) # Exit after a key is pressed
cv2.destroyAllWindows() # Close windows