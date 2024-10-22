import cv2
width = int(input("Enter the width of the rectangle: ")) # User Input
height = int(input("Enter the length of the rectangle: "))
image = cv2.imread('bg.png') # Black BAckground so that rectangle is visible
top_left = (50, 50)
bottom_right = (top_left[0] + width, top_left[1] + height)
color = (255, 255, 51)
thickness = 5 # Thickness of Rectangle
cv2.rectangle(image, top_left, bottom_right, color, thickness)
cv2.imshow("Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()