import cv2
import numpy as np
from updated_script import CombinedDetector
import time

def main():
    # Initialize the detector
    detector = CombinedDetector(enable_sound=True)
    
    # ESP32-CAM stream URL
    stream_url = "http://192.168.50.200:81/stream"
    
    # Create video capture object
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("Error: Could not open ESP32-CAM stream")
        return
    
    print("Successfully connected to ESP32-CAM stream")
    
    try:
        while True:
            # Read frame from stream
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            # Process the frame using the detector
            processed_frame = detector.process_frame(frame)
            
            # Display the processed frame
            cv2.imshow("ESP32-CAM Stream", processed_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping the stream...")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()

if __name__ == "__main__":
    main() 