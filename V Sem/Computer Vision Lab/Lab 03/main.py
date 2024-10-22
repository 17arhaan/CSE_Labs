import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from scipy.ndimage import median_filter, maximum_filter, minimum_filter, gaussian_filter
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
    display_modified_image = ImageTk.PhotoImage(modified_image)
    modified_image_label.config(image=display_modified_image)
    modified_image_label.image = display_modified_image
    plot_histogram(modified_image)  # Update histogram with the modified image

def smooth_image():
    global modified_image
    if original_image:
        try:
            sigma = float(sigma_entry.get())
        except ValueError:
            print("Invalid sigma value")
            return

        image_np = np.array(original_image)
        smoothed_image_np = gaussian_filter(image_np, sigma=sigma)
        modified_image = Image.fromarray(smoothed_image_np.astype(np.uint8))
        update_modified_image()
        plot_kernel(sigma)

def plot_kernel(sigma):
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    x = np.linspace(-sigma, sigma, kernel_size)
    y = np.linspace(-sigma, sigma, kernel_size)
    X, Y = np.meshgrid(x, y)
    gaussian_kernel = np.exp(-0.5 * (X ** 2 + Y ** 2) / sigma ** 2)
    gaussian_kernel /= gaussian_kernel.sum()

    # Create a new top-level window for the kernel plot
    kernel_window = tk.Toplevel(window)
    kernel_window.title(f"Gaussian Kernel (sigma={sigma})")

    figure, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(gaussian_kernel, cmap='viridis', interpolation='nearest')
    figure.colorbar(cax)
    ax.set_title(f"Gaussian Kernel (sigma={sigma})")
    plt.xlabel('X axis')
    plt.ylabel('Y axis')

    canvas = FigureCanvasTkAgg(figure, master=kernel_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def apply_median_filter():
    global modified_image
    if original_image:
        size = int(median_slider.get())
        image_np = np.array(original_image)
        filtered_image_np = median_filter(image_np, size=(size, size, 1))
        modified_image = Image.fromarray(filtered_image_np)
        update_modified_image()

def apply_max_filter():
    global modified_image
    if original_image:
        size = int(max_min_slider.get())
        image_np = np.array(original_image)
        filtered_image_np = maximum_filter(image_np, size=(size, size, 1))
        modified_image = Image.fromarray(filtered_image_np)
        update_modified_image()

def apply_min_filter():
    global modified_image
    if original_image:
        size = int(max_min_slider.get())
        image_np = np.array(original_image)
        filtered_image_np = minimum_filter(image_np, size=(size, size, 1))
        modified_image = Image.fromarray(filtered_image_np)
        update_modified_image()

def sharpen_image():
    global modified_image
    if original_image:
        image_np = np.array(original_image)
        sharpened_image_np = apply_sharpening_matrix(image_np)
        modified_image = Image.fromarray(sharpened_image_np)
        update_modified_image()

def apply_sharpening_matrix(image):
    sharpening_filter = np.array([[0, -0.25, 0], [-0.25, 2, -0.25], [0, -0.25, 0]])
    image = image.astype(np.float32)
    pad_size = 1
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
    sharpened_image = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                sharpened_image[y, x, c] = np.sum(
                    padded_image[y:y + 3, x:x + 3, c] * sharpening_filter
                )
    return np.clip(sharpened_image, 0, 255).astype(np.uint8)

def reset_image():
    global modified_image
    if original_image:
        modified_image = original_image.copy()
        update_modified_image()

def unsharp_mask():
    global modified_image
    if original_image:
        try:
            sigma = float(sigma_entry.get())
            amount = float(amount_entry.get())
        except ValueError:
            print("Invalid sigma or amount value")
            return

        image_np = np.array(original_image)
        blurred_image_np = gaussian_filter(image_np, sigma=sigma)
        sharpened_image_np = image_np + (image_np - blurred_image_np) * amount
        sharpened_image_np = np.clip(sharpened_image_np, 0, 255).astype(np.uint8)
        modified_image = Image.fromarray(sharpened_image_np)
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

def plot_histogram(image):
    figure, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)  # Smaller figure size
    channels = ('Red', 'Green', 'Blue')
    for i, color in enumerate(channels):
        axes[i].hist(np.array(image)[:, :, i].ravel(), bins=256, range=(0, 256), color=color.lower())
        axes[i].set_title(f"{color} Channel")
    for widget in histogram_frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(figure, master=histogram_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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

histogram_frame = tk.Frame(top_frame)
histogram_frame.pack(side=tk.TOP, fill=tk.X, pady=5)  # Add padding between histograms and buttons

select_image_button = tk.Button(button_frame, text="Select Image", command=open_image)
select_image_button.pack(side=tk.LEFT, padx=2, pady=2)

smooth_image_button = tk.Button(button_frame, text="Smooth Image", command=smooth_image)
smooth_image_button.pack(side=tk.LEFT, padx=2, pady=2)

median_filter_button = tk.Button(button_frame, text="Apply Median Filter", command=apply_median_filter)
median_filter_button.pack(side=tk.LEFT, padx=2, pady=2)

max_filter_button = tk.Button(button_frame, text="Apply Max Filter", command=apply_max_filter)
max_filter_button.pack(side=tk.LEFT, padx=2, pady=2)

min_filter_button = tk.Button(button_frame, text="Apply Min Filter", command=apply_min_filter)
min_filter_button.pack(side=tk.LEFT, padx=2, pady=2)

sharpen_button = tk.Button(button_frame, text="Sharpen Image", command=sharpen_image)
sharpen_button.pack(side=tk.LEFT, padx=2, pady=2)

unsharp_button = tk.Button(button_frame, text="Unsharp Mask", command=unsharp_mask)
unsharp_button.pack(side=tk.LEFT, padx=2, pady=2)

reset_button = tk.Button(button_frame, text="Reset Changes", command=reset_image)
reset_button.pack(side=tk.LEFT, padx=2, pady=2)

smoothing_label = tk.Label(slider_frame, text="Smoothing Sigma:")
smoothing_label.pack(side=tk.LEFT, padx=5, pady=5)

sigma_entry = tk.Entry(slider_frame)
sigma_entry.insert(0, "1.0")  # Default sigma value
sigma_entry.pack(side=tk.LEFT, padx=5, pady=5)

amount_label = tk.Label(slider_frame, text="Unsharp Amount:")
amount_label.pack(side=tk.LEFT, padx=5, pady=5)

amount_entry = tk.Entry(slider_frame)
amount_entry.insert(0, "1.0")  # Default amount value
amount_entry.pack(side=tk.LEFT, padx=5, pady=5)

median_filter_label = tk.Label(slider_frame, text="Median Radius:")
median_filter_label.pack(side=tk.LEFT, padx=5, pady=5)

median_slider = tk.Scale(slider_frame, from_=1, to_=21, orient=tk.HORIZONTAL)
median_slider.set(3)
median_slider.pack(side=tk.LEFT, padx=5, pady=5)

max_min_filter_label = tk.Label(slider_frame, text="Max/Min Radius:")
max_min_filter_label.pack(side=tk.LEFT, padx=5, pady=5)

max_min_slider = tk.Scale(slider_frame, from_=1, to_=21, orient=tk.HORIZONTAL)
max_min_slider.set(3)
max_min_slider.pack(side=tk.LEFT, padx=5, pady=5)

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

window.bind("<F11>", toggle_fullscreen)

original_image = None
modified_image = None
display_original_image = None
display_modified_image = None

window.mainloop()