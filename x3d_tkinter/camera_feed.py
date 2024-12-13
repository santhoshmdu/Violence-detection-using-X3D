import cv2
import torch
import torchvision.transforms as transforms
from model_utils import load_model, custome_X3D
import torch.nn.functional as F  # For calculating softmax probabilities

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 2
class_names = ["Violence", "Non Violence"]

model = custome_X3D(num_classes)
model_path = './best_model.pth'
model = load_model(model, model_path, device)
model.eval()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

frame_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read a frame from the camera.")
        break

    frame_resized = cv2.resize(frame, (224, 224))
    frame_buffer.append(frame_resized)

    if len(frame_buffer) > 16:
        frame_buffer.pop(0)

    if len(frame_buffer) == 16:
        # Apply transformation to each frame and stack them
        processed_frames = [transform(frame) for frame in frame_buffer]
        # Stack frames and rearrange dimensions to (B, C, T, H, W)
        input_tensor = torch.stack(processed_frames).permute(1, 0, 2, 3).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            # Get the softmax probabilities for each class
            probabilities = F.softmax(outputs, dim=1)
            
            # Get the predicted class and probability (confidence) for each class
            confidence_violence = probabilities[0][0].item() * 100  # Confidence for 'Violence'
            confidence_non_violence = probabilities[0][1].item() * 100  # Confidence for 'Non Violence'
            
            # Get the predicted class and its label
            _, predicted = torch.max(probabilities, 1)
            label = class_names[predicted.item()]

        # Draw rectangle and labels on the frame with confidence scores for both classes
        cv2.rectangle(frame, (10, 10), (500, 120), (255, 0, 0), -1)
        cv2.putText(frame, f"Violence: {confidence_violence:.2f}%", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Non Violence: {confidence_non_violence:.2f}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Detected: {label}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
