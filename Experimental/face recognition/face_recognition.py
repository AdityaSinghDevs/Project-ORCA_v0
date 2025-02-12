"""
Needs Python version 3.8, 9, 10

"""
<<<<<<< HEAD
import cv2
import numpy as np
import os
import pickle
import time
from deepface import DeepFace
import winsound

=======
import cv2 # type: ignore
import pickle
import numpy as np # type: ignore
from deepface import DeepFace # type: ignore
from scipy.spatial.distance import cosine # type: ignore
>>>>>>> d865dff (WIP: Added Sface and YuNet for face recognition and face detection respectively)

# Load trained classifier and label encoder
with open("face_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Create directory for unknown faces (if not exists)
unknown_folder = "unknown_data"
if not os.path.exists(unknown_folder):
    os.makedirs(unknown_folder)

# Initialize webcam
cap = cv2.VideoCapture(0)

# OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    identified_people = []
    unknown_detected = False

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]  # Extract face region

        # Save face to temp file for DeepFace processing
        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, face_roi)

        try:
            # Extract face embedding
            new_embedding = DeepFace.represent(img_path=temp_path, model_name="Facenet")[0]['embedding']

            # Predict identity
            pred_label = clf.predict([new_embedding])[0]
            pred_name = label_encoder.inverse_transform([pred_label])[0]

            # Predict confidence
            probs = clf.predict_proba([new_embedding])
            confidence = np.max(probs)

            if confidence > 0.4 and pred_name != "Unknown Person":
                label = f"{pred_name} ({confidence:.2f})"
                identified_people.append(pred_name)
                color = (0, 255, 0)  # Green for known people
            else:
                label = "Unknown Person!"
                color = (0, 0, 255)  # Red for unknown person
                unknown_detected = True

                # **Limit to only 5 images in unknown_data folder**
                existing_files = os.listdir(unknown_folder)
                if len(existing_files) < 5:
                    unknown_path = os.path.join(unknown_folder, f"unknown_{int(time.time())}.jpg")
                    cv2.imwrite(unknown_path, face_roi)  # Save unknown face

            # Draw face box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        except Exception as e:
            print(f"Error processing face: {e}")

    # 🚨 **Trigger Alert If One Known + One Unknown Person is in Frame**
    if unknown_detected and len(identified_people) >= 1:
        print("🚨 Unidentified person detected in a group! Alert triggered!")
        winsound.PlaySound("alert_sound.wav", winsound.SND_FILENAME)

    # Show the output
    cv2.imshow("Multi-Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

