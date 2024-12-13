import tkinter as tk
from tkinter import messagebox
import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image, ImageTk
from model_utils import load_model, custome_X3D
import threading
import os
import datetime
import time
import pygame

# Initialize the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and pre-trained weights
num_classes = 2
class_names = ["Violence", "Non Violence"]
model = custome_X3D(num_classes)
model_path = './best_model.pth'
model = load_model(model, model_path, device)
model.eval()

# Initialize sound alert
alert_sound_path = "alert.wav"
pygame.mixer.init()

# Open the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    messagebox.showerror("Error", "Cannot access the camera.")
    exit()

# Set up Tkinter window
window = tk.Tk()
window.title("CCTV Monitoring System")
window.geometry("1000x700")
window.resizable(False, False)

# UI Elements
frame_label = tk.Label(window)
frame_label.grid(row=0, column=0, columnspan=3)

fps_label = tk.Label(window, text="FPS: Calculating...", font=("Helvetica", 12))
fps_label.grid(row=1, column=0, pady=5)

prediction_label = tk.Label(window, text="Detection: Waiting for input...", font=("Helvetica", 16))
prediction_label.grid(row=2, column=0, columnspan=3, pady=10)

control_frame = tk.Frame(window)
control_frame.grid(row=3, column=0, columnspan=3, pady=10)

history_label = tk.Label(window, text="Detection History", font=("Helvetica", 14))
history_label.grid(row=4, column=0, columnspan=3, pady=10)

history_listbox = tk.Listbox(window, height=8, width=50, font=("Helvetica", 10))
history_listbox.grid(row=5, column=0, columnspan=3, pady=10)

footer_label = tk.Label(window, text="© 2024 CCTV Monitoring System. All rights reserved.", font=("Helvetica", 8))
footer_label.grid(row=6, column=0, columnspan=3)

# Variables
frame_buffer = []
history = []
log_file = "detection_log.txt"
alert_triggered = False

# Threshold values
violence_confidence_threshold = 98  # Minimum confidence for "Violence" detection
confidence_gap_threshold = 80      # Minimum gap between Violence and Non Violence confidence

# Transform function for frames
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Functions
def play_alert_sound():
    pygame.mixer.music.load(alert_sound_path)
    pygame.mixer.music.play()

def show_alert_popup():
    messagebox.showwarning("Alert", "Violence Detected with High Confidence!")

def log_detection(entry):
    with open(log_file, "a") as f:
        f.write(f"{entry}\n")

def take_snapshot():
    if len(frame_buffer) > 0:
        frame = frame_buffer[-1]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"snapshot_{timestamp}.png"
        cv2.imwrite(filepath, frame)
        messagebox.showinfo("Snapshot", f"Snapshot saved at {filepath}")

last_time = time.time()

def update_frame():
    global last_time, alert_triggered
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Cannot read a frame from the camera.")
        window.quit()

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - last_time)
    last_time = current_time
    fps_label.config(text=f"FPS: {fps:.2f}")

    # Process Frame
    frame_resized = cv2.resize(frame, (224, 224))
    frame_buffer.append(frame_resized)
    if len(frame_buffer) > 16:
        frame_buffer.pop(0)

    if len(frame_buffer) == 16:
        processed_frames = [transform(f) for f in frame_buffer]
        input_tensor = torch.stack(processed_frames).permute(1, 0, 2, 3).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence_violence = probabilities[0][0].item() * 100
            confidence_non_violence = probabilities[0][1].item() * 100
            _, predicted = torch.max(probabilities, 1)
            label = class_names[predicted.item()]

        prediction_label.config(
            text=f"Violence: {confidence_violence:.2f}% | Non Violence: {confidence_non_violence:.2f}%\nDetected: {label}"
        )

        # Improved Alert System with confidence checks
        if (
            label == "Violence"
            and confidence_violence >= violence_confidence_threshold
            and (confidence_violence - confidence_non_violence) >= confidence_gap_threshold
            and not alert_triggered
        ):
            alert_triggered = True
            # Trigger alert only if the above checks are passed
            threading.Thread(target=play_alert_sound).start()
            threading.Thread(target=show_alert_popup).start()
            log_detection(f"High Confidence Violence Detected: {confidence_violence:.2f}%")
            take_snapshot()

        # Reset alert trigger if the confidence falls below threshold
        if confidence_violence < violence_confidence_threshold or (
            confidence_violence - confidence_non_violence
        ) < confidence_gap_threshold:
            alert_triggered = False

        # Update detection history
        detection_entry = f"{label} detected with {confidence_violence:.2f}% confidence."
        history.append(detection_entry)
        history_listbox.delete(0, tk.END)
        for entry in history[-10:]:  # Show last 10 entries for better performance
            history_listbox.insert(tk.END, entry)

    # Display Frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image=image_pil)
    frame_label.config(image=photo)
    frame_label.image = photo

    window.after(10, update_frame)

# Control Buttons
start_button = tk.Button(control_frame, text="Start Video", command=update_frame)
start_button.grid(row=0, column=0, padx=5)

snapshot_button = tk.Button(control_frame, text="Take Snapshot", command=take_snapshot)
snapshot_button.grid(row=0, column=1, padx=5)

# Run the main loop
window.mainloop()
cap.release()
cv2.destroyAllWindows()
