import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize global variables
original_image = None
modified_image = None
display_original_image = None
display_modified_image = None
harris_threshold = 1.0  # Default value for threshold

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        global original_image, modified_image, display_original_image, display_modified_image
        original_image = Image.open(file_path).convert("RGB")
        
        # Resize image to fit within a reasonable frame size
        max_width = 400  # Adjust as needed
        max_height = 400  # Adjust as needed
        
        width, height = original_image.size
        if width > max_width or height > max_height:
            scaling_factor = min(max_width / width, max_height / height)
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            original_image = original_image.resize(new_size, Image.LANCZOS)
        
        modified_image = original_image.copy()
        display_original_image = ImageTk.PhotoImage(original_image)
        original_image_label.config(image=display_original_image)
        original_image_label.image = display_original_image
        update_modified_image()

def update_modified_image():
    global modified_image, display_modified_image
    if modified_image:
        # Resize image to fit within a reasonable frame size
        max_width = 400  # Adjust as needed
        max_height = 400  # Adjust as needed
        
        width, height = modified_image.size
        if width > max_width or height > max_height:
            scaling_factor = min(max_width / width, max_height / height)
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            modified_image = modified_image.resize(new_size, Image.LANCZOS)
        
        display_modified_image = ImageTk.PhotoImage(modified_image)
        modified_image_label.config(image=display_modified_image)
        modified_image_label.image = display_modified_image

def harris_corner_detection(image, threshold):
    k = 0.04

    # Convert image to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')
    image_np = np.array(image)

    # Compute image gradients
    Ix = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute products of gradients
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    
    # Apply Gaussian filter
    Sx2 = cv2.GaussianBlur(Ix2, (3, 3), 0)
    Sy2 = cv2.GaussianBlur(Iy2, (3, 3), 0)
    Sxy = cv2.GaussianBlur(Ixy, (3, 3), 0)
    
    # Compute Harris response
    det = (Sx2 * Sy2) - (Sxy * Sxy)
    trace = Sx2 + Sy2
    R = det - k * (trace * trace)
    
    # Thresholding
    R_thresholded = np.copy(R)
    R_thresholded[R < (threshold / 10) * R.max()] = 0
    
    # Draw corners
    corners = np.argwhere(R_thresholded > 0)
    
    # Convert image back to color for visualization
    color_image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    for corner in corners:
        y, x = corner
        cv2.circle(color_image_np, (x, y), 5, (0, 255, 0), 1)
    
    return Image.fromarray(color_image_np)

def fast_corner_detection(image, threshold):
    # Convert image to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')
    image_np = np.array(image)
    
    # Initialize FAST detector with threshold adjusted and converted to integer
    fast = cv2.FastFeatureDetector_create(int(threshold * 10))  # Adjust threshold scale and convert to integer
    keypoints = fast.detect(image_np, None)
    
    # Draw keypoints
    image_with_keypoints = cv2.drawKeypoints(image_np, keypoints, None, color=(0, 255, 0))
    
    return Image.fromarray(image_with_keypoints)

def update_threshold(value):
    global harris_threshold
    harris_threshold = float(value)
    if original_image:
        # You can choose to apply both detections or one of them based on your requirement
        modified_image = harris_corner_detection(original_image, harris_threshold)
        update_modified_image()

def harris_corner_detection_button():
    global modified_image
    if original_image:
        modified_image = harris_corner_detection(original_image, harris_threshold)
        update_modified_image()

def fast_corner_detection_button():
    global modified_image
    if original_image:
        modified_image = fast_corner_detection(original_image, harris_threshold)
        update_modified_image()

def update_webcam_feed():
    global live_feed_image, live_feed_label
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        live_feed_image = Image.fromarray(frame_rgb)
        update_live_feed_image()
    else:
        print("Error: Unable to capture image from webcam.")
    live_feed_window.after(10, update_webcam_feed)

def update_live_feed_image():
    global live_feed_image, live_feed_label
    if live_feed_image:
        try:
            live_feed_display_image = ImageTk.PhotoImage(live_feed_image)
            live_feed_label.config(image=live_feed_display_image)
            live_feed_label.image = live_feed_display_image
        except Exception as e:
            print(f"Error: {e}")

def open_live_feed_window():
    global live_feed_window, live_feed_image, live_feed_label
    live_feed_window = tk.Toplevel(window)
    live_feed_window.title("Live Feed")
    live_feed_window.geometry('800x800')  # Adjust size as needed

    live_feed_label = tk.Label(live_feed_window)
    live_feed_label.pack(fill=tk.BOTH, expand=True)

    update_webcam_feed()

def toggle_fullscreen(event=None):
    if window.attributes('-fullscreen'):
        window.attributes('-fullscreen', False)
        window.geometry('1200x800')  # Restore to a default window size
    else:
        window.attributes('-fullscreen', True)

def quit_application(event=None):
    cap.release()
    window.destroy()

def reset_images():
    global original_image, modified_image
    if original_image:
        modified_image = original_image.copy()
        update_modified_image()

window = tk.Tk()
window.title("ComputerVisionApp")
window.geometry('1200x800')  # Set a reasonable window size

top_frame = tk.Frame(window)
top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

middle_frame = tk.Frame(window)
middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

button_frame = tk.Frame(top_frame)
button_frame.pack(side=tk.TOP, fill=tk.X)

# Add buttons
select_image_button = tk.Button(button_frame, text="Select Image", command=open_image)
select_image_button.pack(side=tk.LEFT, padx=2, pady=2)

harris_corner_button = tk.Button(button_frame, text="Harris Corner Detection", command=harris_corner_detection_button)
harris_corner_button.pack(side=tk.LEFT, padx=2, pady=2)

fast_corner_button = tk.Button(button_frame, text="FAST Corner Detection", command=fast_corner_detection_button)
fast_corner_button.pack(side=tk.LEFT, padx=2, pady=2)

open_live_feed_button = tk.Button(button_frame, text="Open Live Feed", command=open_live_feed_window)
open_live_feed_button.pack(side=tk.LEFT, padx=2, pady=2)

reset_button = tk.Button(button_frame, text="Reset Changes", command=reset_images)
reset_button.pack(side=tk.LEFT, padx=2, pady=2)

slider_frame = tk.Frame(top_frame)
slider_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

threshold_label = tk.Label(slider_frame, text="Threshold:")
threshold_label.pack(side=tk.LEFT, padx=5, pady=5)

threshold_slider = tk.Scale(slider_frame, from_=1, to_=10, orient=tk.HORIZONTAL, command=update_threshold)
threshold_slider.set(1)  # Set default threshold value
threshold_slider.pack(side=tk.LEFT, padx=5, pady=5)

input_image_label = tk.Label(middle_frame, text="Input Image")
input_image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

original_image_label = tk.Label(input_image_label)
original_image_label.pack()

output_image_label = tk.Label(middle_frame, text="Output Image")
output_image_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

modified_image_label = tk.Label(output_image_label)
modified_image_label.pack()

window.bind("<F11>", toggle_fullscreen)
window.bind("<Escape>", quit_application)

window.mainloop()
