import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from scipy.ndimage import gaussian_filter
from skimage import feature
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ComputerVisionApp")
        self.root.geometry('1200x800')

        self.original_image = None
        self.modified_image = None

        self.setup_ui()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            self.root.destroy()
            return
        self.update_webcam_feed()

    def setup_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        middle_frame = tk.Frame(self.root)
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        button_frame = tk.Frame(top_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X)

        slider_frame = tk.Frame(top_frame)
        slider_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        webcam_histogram_frame = tk.Frame(top_frame)
        webcam_histogram_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.BOTH, expand=True)

        webcam_frame = tk.Frame(webcam_histogram_frame)
        webcam_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.histogram_frame = tk.Frame(webcam_histogram_frame, width=200, height=150)
        self.histogram_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Adjust webcam label to be 600x600
        self.webcam_label = tk.Label(webcam_frame, width=600, height=600)
        self.webcam_label.pack(side=tk.TOP, padx=5, pady=5, fill=tk.BOTH, expand=True)

        tk.Label(webcam_frame, text="Live Feed", font=("Arial", 14)).pack(side=tk.TOP, padx=5, pady=5)

        tk.Button(button_frame, text="Select Image", command=self.open_image).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(button_frame, text="Canny Edge Detection", command=self.canny_edge_detection).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(button_frame, text="Reset Changes", command=self.reset_image).pack(side=tk.LEFT, padx=2, pady=2)

        tk.Label(slider_frame, text="Threshold:").pack(side=tk.LEFT, padx=5, pady=5)
        self.sigma_entry = tk.Entry(slider_frame)
        self.sigma_entry.insert(0, "1.0")
        self.sigma_entry.pack(side=tk.LEFT, padx=5, pady=5)

        tk.Label(middle_frame, text="Input Image").pack(side=tk.LEFT, padx=10, pady=5)
        tk.Label(middle_frame, text="Output Image").pack(side=tk.RIGHT, padx=10, pady=5)

        self.original_image_label = tk.Label(middle_frame, width=600, height=600)
        self.original_image_label.pack(side=tk.LEFT, padx=10, pady=10, expand=True)

        self.modified_image_label = tk.Label(middle_frame, width=600, height=600)
        self.modified_image_label.pack(side=tk.RIGHT, padx=10, pady=10, expand=True)

        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if file_path:
            self.original_image = Image.open(file_path).convert("RGB")
            self.modified_image = self.original_image.copy()
            self.update_image_labels()
            self.update_modified_image()

    def update_image_labels(self):
        if self.original_image:
            self.original_image_label.config(image=ImageTk.PhotoImage(self.original_image))
            self.original_image_label.image = ImageTk.PhotoImage(self.original_image)
        if self.modified_image:
            self.modified_image_label.config(image=ImageTk.PhotoImage(self.modified_image))
            self.modified_image_label.image = ImageTk.PhotoImage(self.modified_image)

    def update_modified_image(self):
        if self.modified_image:
            self.modified_image_label.config(image=ImageTk.PhotoImage(self.modified_image))
            self.modified_image_label.image = ImageTk.PhotoImage(self.modified_image)
            self.plot_histogram(self.modified_image)

    def canny_edge_detection(self):
        if self.original_image:
            try:
                sigma = float(self.sigma_entry.get())
                image_np = np.array(self.original_image.convert("L"))
                smoothed_image_np = gaussian_filter(image_np, sigma=sigma)
                edges = feature.canny(smoothed_image_np)
                edges_image_np = np.uint8(edges * 255)
                self.modified_image = Image.fromarray(edges_image_np, mode="L")
                self.update_modified_image()
            except ValueError:
                messagebox.showerror("Invalid input", "Please enter a valid number for the threshold.")

    def plot_histogram(self, image):
        figure, axes = plt.subplots(1, 1, figsize=(4, 3))
        axes.hist(np.array(image).ravel(), bins=256, range=(0, 256), color='black')
        axes.set_title("Histogram")

        for widget in self.histogram_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(figure, master=self.histogram_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def reset_image(self):
        if self.original_image:
            self.modified_image = self.original_image.copy()
            self.update_modified_image()

    def update_webcam_feed(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture image")
            self.root.after(10, self.update_webcam_feed)
            return

        # Convert the frame from BGR to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define the color range for segmentation in HSV color space
        lower_blue = np.array([100, 150, 150])
        upper_blue = np.array([140, 255, 255])
        
        # Create a binary mask where the blue color range is within the specified range
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Perform bitwise AND to isolate the blue color in the frame
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours and bounding boxes around detected objects
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert the frame to RGB and resize to 600x600 pixels
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        frame_image = frame_image.resize((600, 600), Image.LANCZOS)
        
        # Convert to PhotoImage and update the label
        frame_photo = ImageTk.PhotoImage(frame_image)
        self.webcam_label.config(image=frame_photo)
        self.webcam_label.image = frame_photo

        # Debugging print statement
        print("Updated webcam feed")

        # Call this function again after 10 milliseconds
        self.root.after(10, self.update_webcam_feed)

    def toggle_fullscreen(self, event=None):
        if self.root.attributes('-fullscreen'):
            self.root.attributes('-fullscreen', False)
            self.root.geometry('1200x800')
        else:
            self.root.attributes('-fullscreen', True)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()