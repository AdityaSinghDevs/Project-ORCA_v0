"""
Needs Python version 3.8, 9, 10

"""
import cv2
import pickle
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Load trained classifier and label encoder
with open("face_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Start Webcam
cap = cv2.VideoCapture(0)  # 0 for built-in webcam, change to 1 for external camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save frame temporarily
    cv2.imwrite("temp_face.jpg", frame)

    try:
        # Extract embedding from the new face
        new_embedding = DeepFace.represent(img_path="temp_face.jpg", model_name="Facenet")[0]['embedding']

        # Predict identity using the trained classifier
        pred_label = clf.predict([new_embedding])[0]
        pred_name = label_encoder.inverse_transform([pred_label])[0]

        # Predict confidence
        probs = clf.predict_proba([new_embedding])
        confidence = np.max(probs)

        # Set a confidence threshold for recognition
        if confidence > 0.4:
            label = f"Recognized: {pred_name} ({confidence:.2f})"
            color = (0, 255, 0)  # Green for recognized
        else:
            label = "Unidentified Identity"
            color = (0, 0, 255)  # Red for unknown

        # Draw text on frame
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Face Recognition", frame)

    except:
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
