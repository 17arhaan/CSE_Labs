import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from scipy.ndimage import gaussian_filter
from skimage import feature
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Initialize the webcam
cap = cv2.VideoCapture(0)

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        global original_image, modified_image, display_original_image, display_modified_image
        original_image = Image.open(file_path).convert("RGB")
        modified_image = original_image.copy()
        display_original_image = ImageTk.PhotoImage(original_image)
        original_image_label.config(image=display_original_image)
        original_image_label.image = display_original_image
        update_modified_image()

def update_modified_image():
    global modified_image, display_modified_image
    display_modified_image = ImageTk.PhotoImage(modified_image)
    modified_image_label.config(image=display_modified_image)
    modified_image_label.image = display_modified_image
    plot_histogram(modified_image)  # Update histogram with the modified image

def canny_edge_detection():
    global modified_image
    if original_image:
        sigma = float(sigma_entry.get())  # Get sigma value for Gaussian smoothing

        image_np = np.array(original_image.convert("L"))  # Convert to grayscale
        smoothed_image_np = gaussian_filter(image_np, sigma=sigma)  # Smooth image
        edges = feature.canny(smoothed_image_np)  # Apply Canny edge detection

        # Convert edges back to an image
        edges_image_np = np.uint8(edges * 255)  # Scale to 0-255
        modified_image = Image.fromarray(edges_image_np, mode="L")
        update_modified_image()

def plot_histogram(image):
    figure, axes = plt.subplots(1, 1, figsize=(4, 3))  # Smaller histogram size
    axes.hist(np.array(image).ravel(), bins=256, range=(0, 256), color='black')
    axes.set_title("Histogram")

    # Clear previous widgets in histogram frame
    for widget in histogram_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(figure, master=histogram_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def reset_image():
    global modified_image
    if original_image:
        modified_image = original_image.copy()
        update_modified_image()

def exit_fullscreen_mode(event=None):
    window.attributes('-fullscreen', False)
    window.geometry('1200x800')  # Ensure this matches the initial window size

def enter_fullscreen_mode(event=None):
    window.attributes('-fullscreen', True)

def toggle_fullscreen(event=None):
    if window.attributes('-fullscreen'):
        exit_fullscreen_mode()
    else:
        enter_fullscreen_mode()

def update_webcam_feed():
    ret, frame = cap.read()
    if ret:
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to an Image object
        frame_image = Image.fromarray(frame_rgb)
        # Convert the Image object to a PhotoImage object
        frame_photo = ImageTk.PhotoImage(frame_image)
        # Update the label with the new image
        webcam_label.config(image=frame_photo)
        webcam_label.image = frame_photo
    # Call this function again after 10 milliseconds
    window.after(10, update_webcam_feed)

window = tk.Tk()
window.title("ComputerVisionApp")
window.geometry('1200x800')  # Set a reasonable window size

top_frame = tk.Frame(window)
top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

middle_frame = tk.Frame(window)
middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

bottom_frame = tk.Frame(window)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

button_frame = tk.Frame(top_frame)
button_frame.pack(side=tk.TOP, fill=tk.X)

slider_frame = tk.Frame(top_frame)
slider_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Move sliders to just below buttons

# Create a frame for the webcam and histogram
webcam_histogram_frame = tk.Frame(top_frame)
webcam_histogram_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.BOTH, expand=True)

webcam_frame = tk.Frame(webcam_histogram_frame)
webcam_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)

histogram_frame = tk.Frame(webcam_histogram_frame, width=200, height=150)  # Set width and height for histogram frame
histogram_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.BOTH, expand=True)

# Add title for webcam feed
webcam_title = tk.Label(webcam_frame, text="Live Feed", font=("Arial", 14))
webcam_title.pack(side=tk.TOP, padx=5, pady=5)

webcam_label = tk.Label(webcam_frame, width=320, height=240)
webcam_label.pack(side=tk.TOP, padx=5, pady=5, fill=tk.BOTH, expand=True)

select_image_button = tk.Button(button_frame, text="Select Image", command=open_image)
select_image_button.pack(side=tk.LEFT, padx=2, pady=2)

canny_edge_button = tk.Button(button_frame, text="Canny Edge Detection", command=canny_edge_detection)
canny_edge_button.pack(side=tk.LEFT, padx=2, pady=2)

reset_button = tk.Button(button_frame, text="Reset Changes", command=reset_image)
reset_button.pack(side=tk.LEFT, padx=2, pady=2)

smoothing_label = tk.Label(slider_frame, text="Threshold:")
smoothing_label.pack(side=tk.LEFT, padx=5, pady=5)

sigma_entry = tk.Entry(slider_frame)
sigma_entry.insert(0, "1.0")  # Default sigma value
sigma_entry.pack(side=tk.LEFT, padx=5, pady=5)

input_image_label = tk.Label(middle_frame, text="Input Image")
input_image_label.pack(side=tk.LEFT, padx=10, pady=5)

output_image_label = tk.Label(middle_frame, text="Output Image")
output_image_label.pack(side=tk.RIGHT, padx=10, pady=5)

original_image_label = tk.Label(middle_frame)
original_image_label.pack(side=tk.LEFT, padx=10, pady=10, expand=True)

modified_image_label = tk.Label(middle_frame)
modified_image_label.pack(side=tk.RIGHT, padx=10, pady=10, expand=True)

original_image_label.config(width=600, height=600)
modified_image_label.config(width=600, height=600)

# Start updating webcam feed
update_webcam_feed()

window.bind("<F11>", toggle_fullscreen)

original_image = None
modified_image = None
display_original_image = None
display_modified_image = None

window.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), window.destroy()))
window.mainloop()