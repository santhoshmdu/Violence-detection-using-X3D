import tkinter as tk
from tkinter import messagebox
import cv2
import torch
from torchvision import transforms

import torch.nn.functional as F
from PIL import Image, ImageTk
from model_utils import load_model, custome_X3D

# Initialize the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and pre-trained weights
num_classes = 2
class_names = ["Violence", "Non Violence"]
model = custome_X3D(num_classes)
model_path = './best_model.pth'
model = load_model(model, model_path, device)
model.eval()

# Open the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    messagebox.showerror("Error", "Cannot access the camera.")
    exit()

# Set up Tkinter window
window = tk.Tk()
window.title("CCTV Monitoring System")
window.geometry("800x600")
window.resizable(False, False)

# UI Elements
frame_label = tk.Label(window)
frame_label.pack()

prediction_label = tk.Label(window, text="Detection: Waiting for input...", font=("Helvetica", 16))
prediction_label.pack(pady=10)

control_frame = tk.Frame(window)
control_frame.pack(pady=10)

history_label = tk.Label(window, text="Detection History", font=("Helvetica", 14))
history_label.pack(pady=10)

history_listbox = tk.Listbox(window, height=6, width=50, font=("Helvetica", 10))
history_listbox.pack(pady=10)

footer_label = tk.Label(window, text="© 2024 CCTV Monitoring System. All rights reserved.", font=("Helvetica", 8), anchor="center")
footer_label.pack(side="bottom", fill="x", pady=5)

# Video processing and detection functionality
frame_buffer = []
history = []

# Transform function for frames
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Later in your code, you can use `transform` to process frames
processed_frames = [transform(f) for f in frame_buffer]

def update_frame():
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Cannot read a frame from the camera.")
        window.quit()

    frame_resized = cv2.resize(frame, (224, 224))
    frame_buffer.append(frame_resized)

    if len(frame_buffer) > 16:
        frame_buffer.pop(0)

    if len(frame_buffer) == 16:
        # Apply transformation to frames
        processed_frames = [transform(f) for f in frame_buffer]
        input_tensor = torch.stack(processed_frames).permute(1, 0, 2, 3).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            confidence_violence = probabilities[0][0].item() * 100
            confidence_non_violence = probabilities[0][1].item() * 100
            _, predicted = torch.max(probabilities, 1)
            label = class_names[predicted.item()]

        prediction_label.config(text=f"Violence: {confidence_violence:.2f}% | Non Violence: {confidence_non_violence:.2f}%\nDetected: {label}")

        # Update detection history
        history.append(f"{label} detected with {confidence_violence:.2f}% Violence confidence.")
        history_listbox.delete(0, tk.END)
        for entry in history:
            history_listbox.insert(tk.END, entry)

    # Convert frame for Tkinter display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image=image_pil)
    
    frame_label.config(image=photo)
    frame_label.image = photo

    window.after(10, update_frame)

def start_video():
    cap.open(0, cv2.CAP_DSHOW)
    update_frame()

def stop_video():
    cap.release()
    window.quit()

def toggle_detection():
    if prediction_label.cget("text").startswith("Detection: Waiting"):
        start_video()
    else:
        stop_video()

# Add control buttons
start_button = tk.Button(control_frame, text="Start Video", command=start_video)
start_button.pack(side="left", padx=5)

stop_button = tk.Button(control_frame, text="Stop Video", command=stop_video)
stop_button.pack(side="left", padx=5)

toggle_button = tk.Button(control_frame, text="Toggle Detection", command=toggle_detection)
toggle_button.pack(side="left", padx=5)

# Run the Tkinter main loop
window.mainloop()

# Release the webcam after the app closes
cap.release()
cv2.destroyAllWindows()
