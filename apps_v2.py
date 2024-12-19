import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Road Defect Detection")
        self.root.geometry("1280x720")
        self.root.resizable(False, False)  # Fix window size

        # Initialize interface first
        self.setup_interface()

        # Load the YOLO model
        self.model = self.load_model("good_models/v1.pt")

    def setup_interface(self):
        # Left panel
        self.left_panel = tk.Frame(self.root, width=200, bg="#2C3E50")
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.left_panel, text="Navigation", bg="#2C3E50", fg="white", font=("Arial", 14)).pack(pady=10)

        self.btn_page_image = tk.Button(self.left_panel, text="Image Processing", command=self.show_image_page, bg="#34495E", fg="white", relief="flat")
        self.btn_page_image.pack(pady=5, fill=tk.X)

        self.btn_page_video = tk.Button(self.left_panel, text="Video Processing", command=self.show_video_page, bg="#34495E", fg="white", relief="flat")
        self.btn_page_video.pack(pady=5, fill=tk.X)

        self.btn_page_camera = tk.Button(self.left_panel, text="Camera Capture", command=self.show_camera_page, bg="#34495E", fg="white", relief="flat")
        self.btn_page_camera.pack(pady=5, fill=tk.X)

        # Main section
        self.main_section = tk.Frame(self.root, bg="white")
        self.main_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.top_controls = tk.Frame(self.main_section, bg="white", height=50)
        self.top_controls.pack(side=tk.TOP, fill=tk.X)

        self.visualization_area = tk.Frame(self.main_section, bg="black")
        self.visualization_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.console = tk.Text(self.main_section, height=5, bg="#17202A", fg="white", state="disabled", wrap=tk.WORD)
        self.console.pack(side=tk.BOTTOM, fill=tk.X)

        # Toggles
        self.save_results = tk.BooleanVar(value=False)
        self.process_video_toggle = tk.BooleanVar(value=False)

        self.btn_toggle_save = tk.Checkbutton(
            self.top_controls, text="Save Results", variable=self.save_results, onvalue=True, offvalue=False, bg="white"
        )
        self.btn_toggle_save.pack(side=tk.LEFT, padx=5)

        self.btn_toggle_process = tk.Checkbutton(
            self.top_controls, text="Process Video in Real-Time", variable=self.process_video_toggle, onvalue=True, offvalue=False, bg="white"
        )
        self.btn_toggle_process.pack(side=tk.LEFT, padx=5)

        self.current_page = None

    def log(self, message):
        self.console.config(state="normal")
        self.console.insert(tk.END, f"{message}\n")
        self.console.see(tk.END)
        self.console.config(state="disabled")

    def load_model(self, model_path):
        try:
            self.log(f"Loading model from {model_path}...")
            model = YOLO(model_path)  # Load model using ultralytics
            self.log(f"Model {model_path} loaded successfully.")
            return model
        except Exception as e:
            self.log(f"Error loading model: {e}")
            raise

    def show_image_page(self):
        self.clear_visualization_area()
        tk.Label(self.visualization_area, text="Image Processing", bg="black", fg="white", font=("Arial", 18)).pack(pady=20)
        tk.Button(self.visualization_area, text="Upload Image", command=self.upload_image, bg="#5D6D7E", fg="white", relief="flat").pack(pady=10)

    def show_video_page(self):
        self.clear_visualization_area()
        tk.Label(self.visualization_area, text="Video Processing", bg="black", fg="white", font=("Arial", 18)).pack(pady=20)
        tk.Button(self.visualization_area, text="Upload Video", command=self.upload_video, bg="#5D6D7E", fg="white", relief="flat").pack(pady=10)

    def show_camera_page(self):
        self.clear_visualization_area()
        tk.Label(self.visualization_area, text="Camera Capture", bg="black", fg="white", font=("Arial", 18)).pack(pady=20)
        tk.Button(self.visualization_area, text="Start Capture", command=self.capture_from_camera, bg="#5D6D7E", fg="white", relief="flat").pack(pady=10)

    def clear_visualization_area(self):
        for widget in self.visualization_area.winfo_children():
            widget.destroy()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not file_path:
            return
        self.log(f"Image uploaded: {file_path}")
        self.process_image(file_path)

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
        if not file_path:
            return
        self.log(f"Video uploaded: {file_path}")
        self.process_video(file_path)

    def capture_from_camera(self):
        self.log("Starting camera capture...")
        # Placeholder for camera capture functionality

    def process_image(self, file_path):
        try:
            image = Image.open(file_path)
            image = image.resize((640, 640))  # Resize to 640x640
            processed_image = self.run_model_on_image(image)
            self.display_image(processed_image)

        except Exception as e:
            self.log(f"Error processing image: {e}")

    def process_video(self, file_path):
        try:
            capture = cv2.VideoCapture(file_path)
            if not capture.isOpened():
                raise ValueError("Cannot open video file.")

            if self.process_video_toggle.get():
                self.log("Processing video in real-time...")
                while capture.isOpened():
                    ret, frame = capture.read()
                    if not ret:
                        break

                    frame = cv2.resize(frame, (640, 640))  # Resize frame
                    processed_frame = self.run_model_on_frame(frame)
                    self.display_video_frame(processed_frame)
                    self.root.update()

            else:
                self.log("Processing entire video...")
                processed_frames = []
                while capture.isOpened():
                    ret, frame = capture.read()
                    if not ret:
                        break

                    frame = cv2.resize(frame, (640, 640))  # Resize frame
                    processed_frames.append(self.run_model_on_frame(frame))

                self.display_entire_video(processed_frames)

            capture.release()

        except Exception as e:
            self.log(f"Error processing video: {e}")

    def run_model_on_image(self, image):
        try:
            self.log("Running model on image...")
            results = self.model(image)  # Run YOLO model inference
            processed_image = results[0].plot()  # Draw results on image
            return Image.fromarray(processed_image)
        except Exception as e:
            self.log(f"Error during image processing: {e}")
            return image

    def run_model_on_frame(self, frame):
        try:
            self.log("Running model on video frame...")
            results = self.model(frame)  # Run YOLO model inference
            processed_frame = results[0].plot()  # Draw results on frame
            return processed_frame
        except Exception as e:
            self.log(f"Error during frame processing: {e}")
            return frame

    def display_image(self, processed):
        if self.visualization_area.winfo_children():
            self.clear_visualization_area()

        processed = ImageTk.PhotoImage(processed)

        processed_label = tk.Label(self.visualization_area, image=processed, bg="black")
        processed_label.image = processed  # Keep a reference
        processed_label.pack(expand=True)

    def display_video_frame(self, frame):
        # Placeholder for video frame display
        pass

    def display_entire_video(self, frames):
        # Placeholder for displaying entire processed video
        self.log("Displaying processed video... (not implemented yet)")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
