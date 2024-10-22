import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps, ImageDraw
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

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
    image_np = np.array(modified_image)
    gamma_value = gamma_slider.get()
    processed_image = apply_gamma_correction(image_np, gamma_value)
    modified_image = Image.fromarray(processed_image)
    display_modified_image = ImageTk.PhotoImage(modified_image)
    modified_image_label.config(image=display_modified_image)
    modified_image_label.image = display_modified_image
    plot_histogram(modified_image)

def invert_image():
    global modified_image
    if original_image:
        modified_image = ImageOps.invert(original_image)
        update_modified_image()

def apply_log_transform():
    if original_image:
        image_np = np.array(original_image).astype(float)
        log_transformed_image = log_transform(image_np)
        modified_image = Image.fromarray(log_transformed_image)
        update_modified_image()

def find_brightest_region():
    if original_image:
        image_np = np.array(original_image)
        brightest_x, brightest_y = get_brightest_region(image_np)
        draw_circle_on_image(brightest_x, brightest_y)
        update_modified_image()

def apply_gamma_correction(image, gamma):
    gamma = float(gamma)
    inverse_gamma = 1.0 / gamma
    lookup_table = np.array([((i / 255.0) ** inverse_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, lookup_table)

def log_transform(image):
    constant = 255 / np.log(1 + np.max(image))
    log_transformed_image = np.zeros_like(image, dtype=float)
    for i in range(3):
        log_transformed_image[:, :, i] = constant * np.log(1 + image[:, :, i])
    return np.clip(log_transformed_image, 0, 255).astype(np.uint8)

def get_brightest_region(image):
    gray_image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    y, x = np.unravel_index(np.argmax(gray_image), gray_image.shape)
    return x, y

def draw_circle_on_image(x, y, radius=10, color='red'):
    global modified_image
    if modified_image:
        draw = ImageDraw.Draw(modified_image)
        left_up = (x - radius, y - radius)
        right_down = (x + radius, y + radius)
        draw.ellipse([left_up, right_down], outline=color, width=3)

def plot_histogram(image):
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    channels = ('Red', 'Green', 'Blue')
    for i, color in enumerate(channels):
        axes[i].hist(np.array(image)[:, :, i].ravel(), bins=256, range=(0, 256), color=color.lower())
        axes[i].set_title(f"{color} Channel")
        axes[i].set_xlabel('Pixel Value')
        axes[i].set_ylabel('Frequency')
    for widget in histogram_frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(figure, master=histogram_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

def reset_image():
    global modified_image
    if original_image:
        modified_image = original_image.copy()
        update_modified_image()

def exit_fullscreen_mode():
    window.attributes('-fullscreen', False)
    window.geometry('800x600')

window = tk.Tk()
window.title("ComputerVisionApp")
window.attributes('-fullscreen', True)

# Create the top frame for buttons
top_frame = tk.Frame(window)
top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

# Create the middle frame for images
middle_frame = tk.Frame(window)
middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Create the bottom frame for histograms
bottom_frame = tk.Frame(window)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

# Buttons and Sliders in top frame
select_image_button = tk.Button(top_frame, text="Select Image", command=open_image)
select_image_button.pack(side=tk.LEFT, padx=5, pady=5)

invert_image_button = tk.Button(top_frame, text="Negative Image", command=invert_image)
invert_image_button.pack(side=tk.LEFT, padx=5, pady=5)

log_transform_button = tk.Button(top_frame, text="Apply Log Transform", command=apply_log_transform)
log_transform_button.pack(side=tk.LEFT, padx=5, pady=5)

find_brightest_button = tk.Button(top_frame, text="Find Brightest Spot", command=find_brightest_region)
find_brightest_button.pack(side=tk.LEFT, padx=5, pady=5)

reset_button = tk.Button(top_frame, text="Reset Changes", command=reset_image)
reset_button.pack(side=tk.LEFT, padx=5, pady=5)

gamma_label = tk.Label(top_frame, text="Apply Gamma:")
gamma_label.pack(side=tk.LEFT, padx=5, pady=5)

gamma_slider = tk.Scale(top_frame, from_=0.1, to_=5.0, orient=tk.HORIZONTAL, resolution=0.1, command=lambda x: update_modified_image())
gamma_slider.set(1.0)
gamma_slider.pack(side=tk.LEFT, padx=5, pady=5)

exit_fullscreen_button = tk.Button(top_frame, text="Exit Full Screen", command=exit_fullscreen_mode)
exit_fullscreen_button.pack(side=tk.RIGHT, padx=5, pady=5)

# Labels for images in the middle frame
input_image_label = tk.Label(middle_frame, text="Input Image")
input_image_label.pack(side=tk.LEFT, padx=10, pady=5)

output_image_label = tk.Label(middle_frame, text="Output Image")
output_image_label.pack(side=tk.RIGHT, padx=10, pady=5)

# Image Labels in the middle frame
original_image_label = tk.Label(middle_frame)
original_image_label.pack(side=tk.LEFT, padx=10, pady=10, expand=True)

modified_image_label = tk.Label(middle_frame)
modified_image_label.pack(side=tk.RIGHT, padx=10, pady=10, expand=True)

# Histogram frame
histogram_frame = tk.Frame(bottom_frame)
histogram_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

# Initialize variables
original_image = None
modified_image = None
display_original_image = None
display_modified_image = None

window.mainloop()
