import cv2
import numpy as np
import torch
import time
import math
import json
import os
import contextlib
from ultralytics import YOLOv10

class DistanceEstimator:
    def __init__(self):
        # Load YOLOv10 model from specified repository
        self.model = YOLOv10.from_pretrained('jameslahm/yolov10n')  # Use the specified YOLOv10 model
        
        # Known parameters (these will need calibration for accurate results)
        self.KNOWN_WIDTH = {
            'person': 60,    # average width of a person in cm
            'car': 180,      # average width of a car in cm
            'bottle': 8,     # average width of a bottle in cm
            'chair': 50,     # average width of a chair in cm
            'laptop': 35,    # average width of a laptop in cm
            'cell phone': 7, # average width of a cell phone in cm
            # Add more objects as needed
        }
        
        # Default width for objects not in the known list
        self.DEFAULT_WIDTH = 30  # cm
        
        # Calibration: focal length in pixels
        # This needs to be calibrated for your specific camera
        self.FOCAL_LENGTH = None
        self.KNOWN_DISTANCE = 100  # cm, distance used for calibration
        self.CALIBRATION_OBJECT = 'bottle'  # Object used for calibration
        
        # Colors for visualization
        self.COLORS = np.random.uniform(0, 255, size=(80, 3))
        
        # Verbose mode flag
        self.verbose_mode = False
        
        # Model verbose mode flag (set to False to disable model prints)
        self.model_verbose = False
    
    def calibrate_camera(self, frame, detections):
        """Calibrate the camera using a known object at a known distance"""
        for detection in detections:
            # Check if there are any detections
            if len(detection.boxes) == 0:
                continue
                
            # Check if cls field exists and has elements
            if not hasattr(detection.boxes, 'cls') or detection.boxes.cls.size(0) == 0:
                continue
                
            class_id = int(detection.boxes.cls.cpu().numpy()[0])
            class_name = self.model.names[class_id]
            
            if class_name == self.CALIBRATION_OBJECT:
                bbox = detection.boxes.xyxy.cpu().numpy()[0]
                object_width_pixels = bbox[2] - bbox[0]
                
                # Calculate focal length
                self.FOCAL_LENGTH = (object_width_pixels * self.KNOWN_DISTANCE) / self.KNOWN_WIDTH.get(class_name, self.DEFAULT_WIDTH)
                if self.verbose_mode:
                    print(f"Camera calibrated! Focal length: {self.FOCAL_LENGTH}")
                return True
        
        return False
    
    def estimate_distance(self, object_width_pixels, class_name):
        """Estimate distance based on object width in pixels"""
        if self.FOCAL_LENGTH is None:
            return None
        
        # Get the known width for this object class
        known_width = self.KNOWN_WIDTH.get(class_name, self.DEFAULT_WIDTH)
        
        # Calculate distance
        distance = (known_width * self.FOCAL_LENGTH) / object_width_pixels
        return distance
    
    def process_frame(self, frame, calibration_mode=False):
        """Process a frame to detect objects and estimate distances"""
        # Convert frame to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLOv10 inference - with verbose turned off to prevent console spam
        try:
            # Redirect stdout to null device if we don't want model output
            if not self.model_verbose:
                with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                    results = self.model(frame_rgb, verbose = False)
            else:
                results = self.model(frame_rgb)
        except Exception as e:
            print(f"Error during inference: {e}")
            return frame  # Return original frame if inference fails
        
        # If in calibration mode, try to calibrate the camera
        if calibration_mode and self.FOCAL_LENGTH is None:
            self.calibrate_camera(frame, results)
            if self.FOCAL_LENGTH is not None and self.verbose_mode:
                print(f"Hold the {self.CALIBRATION_OBJECT} at {self.KNOWN_DISTANCE}cm from camera. Press 'c' when ready.")
        
        # Draw bounding boxes and distances
        annotated_frame = frame.copy()
        
        for detection in results:
            # Skip if no detections
            if len(detection.boxes) == 0:
                continue
                
            # Get detection data, handling potential empty arrays
            try:
                boxes = detection.boxes.xyxy.cpu().numpy()
                scores = detection.boxes.conf.cpu().numpy()
                class_ids = detection.boxes.cls.cpu().numpy().astype(int)
            except (IndexError, AttributeError):
                continue
            
            for i, box in enumerate(boxes):
                class_id = class_ids[i]
                class_name = self.model.names[class_id]
                confidence = scores[i]
                
                # Only process high-confidence detections
                if confidence < 0.5:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                
                # Calculate width in pixels
                object_width_pixels = x2 - x1
                
                # Get object color
                color = self.COLORS[class_id % len(self.COLORS)]
                color = tuple(map(int, color))
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Estimate distance if camera is calibrated
                distance_text = "Unknown"
                if self.FOCAL_LENGTH is not None:
                    distance = self.estimate_distance(object_width_pixels, class_name)
                    if distance is not None:
                        distance_text = f"{distance:.2f} cm"
                
                # Create label with class name, confidence and distance
                label = f"{class_name}: {confidence:.2f}, Dist: {distance_text}"
                
                # Calculate label position
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y1 = max(y1, label_size[1])
                
                # Draw label background
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated_frame
    
    def toggle_verbose(self):
        """Toggle verbose output mode"""
        self.verbose_mode = not self.verbose_mode
        print(f"Verbose mode {'ON' if self.verbose_mode else 'OFF'}")
    
    def toggle_model_verbose(self):
        """Toggle model output verbose mode"""
        self.model_verbose = not self.model_verbose
        print(f"Model verbose mode {'ON' if self.model_verbose else 'OFF'}")
    
    def save_calibration(self, filename="camera_calibration.json"):
        """Save camera calibration data to file"""
        if self.FOCAL_LENGTH is None:
            print("No calibration data to save. Please calibrate the camera first.")
            return False
            
        calibration_data = {
            "focal_length": self.FOCAL_LENGTH,
            "calibration_object": self.CALIBRATION_OBJECT,
            "known_distance": self.KNOWN_DISTANCE,
            "known_widths": self.KNOWN_WIDTH
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=4)
            print(f"Calibration saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False

    def load_calibration(self, filename="camera_calibration.json"):
        """Load camera calibration data from file"""
        try:
            with open(filename, 'r') as f:
                calibration_data = json.load(f)
                
            self.FOCAL_LENGTH = calibration_data.get("focal_length")
            self.CALIBRATION_OBJECT = calibration_data.get("calibration_object", self.CALIBRATION_OBJECT)
            self.KNOWN_DISTANCE = calibration_data.get("known_distance", self.KNOWN_DISTANCE)
            loaded_widths = calibration_data.get("known_widths", {})
            # Update known widths if they exist in the file
            if loaded_widths:
                self.KNOWN_WIDTH.update(loaded_widths)
            
            print(f"Calibration loaded from {filename}")
            print(f"Focal length: {self.FOCAL_LENGTH}")
            return True
        except FileNotFoundError:
            print(f"Calibration file {filename} not found.")
            return False
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set resolution (adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 768)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 432)
    
    # Initialize the distance estimator
    estimator = DistanceEstimator()
    
    # Try to load previous calibration
    estimator.load_calibration()
    
    # Calibration mode flag
    calibration_mode = False
    
    print("=== Object Distance Estimation ===")
    print("Press 'c' to toggle calibration mode.")
    print(f"Hold the {estimator.CALIBRATION_OBJECT} at {estimator.KNOWN_DISTANCE}cm from camera for calibration.")
    print("Press 's' to save calibration.")
    print("Press 'l' to load saved calibration.")
    print("Press 'v' to toggle verbose mode.")
    print("Press 'm' to toggle model output (detection logs).")
    print("Press 'q' to quit.")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Process the frame
        processed_frame = estimator.process_frame(frame, calibration_mode)
        
        # Add calibration status to display
        status_text = f"Calibration: {'READY' if estimator.FOCAL_LENGTH is not None else 'NOT CALIBRATED'}"
        if calibration_mode:
            status_text += " | CALIBRATION MODE ACTIVE"
            
        cv2.putText(processed_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # Display the processed frame
        cv2.imshow("Object Distance Estimation", processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Quit
            break
        elif key == ord('c'):
            # Toggle calibration mode
            calibration_mode = not calibration_mode
            if calibration_mode:
                # Reset calibration to recalibrate
                print("Calibration mode ON.")
                if estimator.FOCAL_LENGTH is not None:
                    choice = input("Reset current calibration? (y/n): ")
                    if choice.lower() == 'y':
                        estimator.FOCAL_LENGTH = None
                        print(f"Calibration reset. Hold the {estimator.CALIBRATION_OBJECT} at {estimator.KNOWN_DISTANCE}cm from camera.")
                else:
                    print(f"Hold the {estimator.CALIBRATION_OBJECT} at {estimator.KNOWN_DISTANCE}cm from camera.")
            else:
                print("Calibration mode OFF.")
        elif key == ord('s'):
            # Save calibration
            estimator.save_calibration()
        elif key == ord('l'):
            # Load calibration
            estimator.load_calibration()
        elif key == ord('v'):
            # Toggle verbose mode
            estimator.toggle_verbose()
        elif key == ord('m'):
            # Toggle model verbose mode
            estimator.toggle_model_verbose()
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()