import cv2
cap = cv2.VideoCapture('6981614-hd_1920_1080_30fps.mp4') # Read Video
while True:
    ret, vid = cap.read() # Define Frame
    cv2.imshow('Video', vid) # sHOW vIDEO
    if cv2.waitKey(1) & 0xFF == ord('q'): # Run whole video or exit when 'q' is pressed
        break
cap.release() # Release the vid object
cv2.destroyAllWindows() # Close Windows