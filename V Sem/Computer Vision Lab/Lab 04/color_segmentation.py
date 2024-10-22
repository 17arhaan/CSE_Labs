import cv2
import numpy as np

# Predefined HSV values for different colors
COLOR_HSV_VALUES = {
    'red': (np.array([0, 50, 50]), np.array([10, 255, 255])),
    'green': (np.array([35, 50, 50]), np.array([85, 255, 255])),
    'blue': (np.array([100, 150, 150]), np.array([140, 255, 255])),
    'yellow': (np.array([20, 100, 100]), np.array([30, 255, 255])),
    'orange': (np.array([10, 100, 100]), np.array([20, 255, 255])),
    'purple': (np.array([130, 50, 50]), np.array([160, 255, 255])),
    'cyan': (np.array([85, 50, 50]), np.array([100, 255, 255])),
    'magenta': (np.array([140, 50, 50]), np.array([160, 255, 255])),
    'pink': (np.array([160, 50, 50]), np.array([170, 255, 255])),
    'brown': (np.array([10, 50, 50]), np.array([20, 150, 150])),
    'gray': (np.array([0, 0, 50]), np.array([180, 30, 200])),
    'black': (np.array([0, 0, 0]), np.array([180, 255, 50])),
    'white': (np.array([0, 0, 200]), np.array([180, 50, 255])),
}

def get_hsv_from_dict(color_name):
    color_name = color_name.lower()
    return COLOR_HSV_VALUES.get(color_name, (None, None))

def main():
    # Get HSV color range from predefined values based on user input
    color_name = input("Color ? :  ")
    lower_bound, upper_bound = get_hsv_from_dict(color_name)
    
    if lower_bound is None or upper_bound is None:
        print("Color not found in the predefined list. Exiting.")
        return
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame from BGR to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a binary mask where the color range is within the specified range
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Perform bitwise AND to isolate the color in the frame
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours and bounding boxes around detected objects
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter out small contours
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box in green

        # Display the original frame with detected objects
        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()