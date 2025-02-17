#requirements:
#pip install git+https://github.com/THU-MIG/yolov10.git
#pip install huggingface_hub



from ultralytics import YOLOv10
import cv2  

# Load the YOLOv10n model  
model = YOLOv10.from_pretrained('jameslahm/yolov10n')

# Open the camera (0 is the default webcam)  
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

    # Perform inference on the frame  
    results = model.predict(source=frame)  

    # Draw results on the frame  
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]  # Get object label
            confidence = box.conf[0]  # Confidence score

            # Draw bounding box  
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label text  
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame  
    cv2.imshow('YOLOv10 Live Detection', frame)  

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit  
        break  

cap.release()
cv2.destroyAllWindows()
