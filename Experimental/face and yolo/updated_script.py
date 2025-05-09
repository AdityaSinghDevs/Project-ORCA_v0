
import os
import sys
import glob
import time
import math
import cv2
import numpy as np
import shutil
import platform
import json
import contextlib
from tqdm import tqdm
from datetime import datetime
from ultralytics import YOLOv10

class CombinedDetector:
    def __init__(self, enable_sound=True, calibration_file="camera_calibration.json"):
        self.directory = 'data'
        self.COSINE_THRESHOLD = 0.5
        self.temp_unknown_dir = os.path.join(self.directory, 'temp_unknown_faces')
        self.enable_sound = enable_sound
        
        # Known widths for distance estimation
        self.KNOWN_WIDTH = {
            'person': 60,
            'car': 180,
            'bottle': 8,
            'chair': 50,
            'laptop': 35,
            'cell phone': 7,
        }
        self.DEFAULT_WIDTH = 30  # cm
        
        # Load calibration data
        self.load_calibration(calibration_file)
        
        # Setup components
        if self.enable_sound:
            self.setup_sound()
        self.setup_face_detection()
        self.setup_object_detection()
        self.setup_temp_directory()
        self.load_face_dictionary()
        self.next_unknown_id = 1
        self.detected_unknowns = set()
        
        # Colors for object visualization
        self.COLORS = np.random.uniform(0, 255, size=(80, 3))
        
        # Verbose mode flags
        self.verbose_mode = False
        self.model_verbose = False

    def setup_sound(self):
        """Setup sound based on platform"""
        self.system = platform.system()
        if self.system == 'Windows':
            try:
                import winsound
                self.sound_function = lambda: winsound.Beep(1000, 500)
                self.sound_available = True
            except Exception as e:
                print(f"Warning: Could not initialize Windows sound: {e}")
                self.sound_available = False
        elif self.system == 'Darwin':
            self.sound_function = lambda: os.system('afplay /System/Library/Sounds/Ping.aiff')
            self.sound_available = True
        elif self.system == 'Linux':
            self.sound_function = lambda: print('\a', flush=True)
            self.sound_available = True
        else:
            print(f"Warning: Sound not supported on {self.system}")
            self.sound_available = False

    def play_notification(self):
        """Safely play notification sound if enabled and available"""
        if self.enable_sound and self.sound_available:
            try:
                self.sound_function()
            except Exception as e:
                print(f"Warning: Could not play sound: {e}")
                self.sound_available = False

    def setup_temp_directory(self):
        """Create or clean temporary directory for unknown faces"""
        if os.path.exists(self.temp_unknown_dir):
            shutil.rmtree(self.temp_unknown_dir)
        os.makedirs(self.temp_unknown_dir)

    def setup_face_detection(self):
        """Initialize face detection and recognition models"""
        weights = os.path.join(self.directory, "models", "face_detection_yunet_2023mar.onnx")
        self.face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
        self.face_detector.setScoreThreshold(0.87)

        weights = os.path.join(self.directory, "models", "face_recognition_sface_2021dec_int8bq.onnx")
        self.face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    def setup_object_detection(self):
        """Initialize YOLOv10 model for object detection"""
        self.yolo_model = YOLOv10.from_pretrained('jameslahm/yolov10n')

    def load_face_dictionary(self):
        """Load registered faces from image files"""
        self.dictionary = {}
        types = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG')
        files = []
        for a_type in types:
            files.extend(glob.glob(os.path.join(self.directory, 'images', a_type)))
        files = list(set(files))

        for file in tqdm(files, desc="Loading registered faces"):
            image = cv2.imread(file)
            feats, faces, _ = self.recognize_face(image, file)
            if faces is None:
                continue
            user_id = os.path.splitext(os.path.basename(file))[0]
            self.dictionary[user_id] = feats[0]

        print(f'Total {len(self.dictionary)} registered IDs loaded')

    def load_calibration(self, filename="camera_calibration.json"):
        """Load camera calibration data from file"""
        try:
            with open(filename, 'r') as f:
                calibration_data = json.load(f)
                
            self.FOCAL_LENGTH = calibration_data.get("focal_length")
            loaded_widths = calibration_data.get("known_widths", {})
            if loaded_widths:
                self.KNOWN_WIDTH.update(loaded_widths)
            
            print(f"Calibration loaded from {filename}")
            print(f"Focal length: {self.FOCAL_LENGTH}")
            return True
        except FileNotFoundError:
            print(f"Calibration file {filename} not found.")
            self.FOCAL_LENGTH = None
            return False
        except Exception as e:
            print(f"Error loading calibration: {e}")
            self.FOCAL_LENGTH = None
            return False

    def match(self, feature1):
        """Match face feature against dictionary"""
        max_score = 0.0
        sim_user_id = ""
        for user_id, feature2 in zip(self.dictionary.keys(), self.dictionary.values()):
            score = self.face_recognizer.match(
                feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
            if score >= max_score:
                max_score = score
                sim_user_id = user_id
        if max_score < self.COSINE_THRESHOLD:
            return False, ("", 0.0)
        return True, (sim_user_id, max_score)

    def recognize_face(self, image, file_name=None):
        """Perform face detection and feature extraction"""
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if image.shape[0] > 1000:
            image = cv2.resize(image, (0, 0),
                             fx=500 / image.shape[0], fy=500 / image.shape[0])

        height, width, _ = image.shape
        self.face_detector.setInputSize((width, height))
        try:
            _, faces = self.face_detector.detect(image)
            if file_name is not None:
                assert len(faces) > 0, f'the file {file_name} has no face'

            faces = faces if faces is not None else []
            features = []
            aligned_face = None
            for face in faces:
                aligned_face = self.face_recognizer.alignCrop(image, face)
                feat = self.face_recognizer.feature(aligned_face)
                features.append(feat)
            return features, faces, aligned_face
        except Exception as e:
            print(e)
            print(file_name)
            return None, None, None

    def save_unknown_face(self, image, face_box, aligned_face, features, unknown_id):
        """Save unknown face image and features"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"unknown_{unknown_id}_{timestamp}.jpg"
        image_path = os.path.join(self.temp_unknown_dir, image_filename)
        cv2.imwrite(image_path, aligned_face)
        
        feature_filename = f"unknown_{unknown_id}_{timestamp}.npy"
        feature_path = os.path.join(self.temp_unknown_dir, feature_filename)
        np.save(feature_path, features[0])
        
        return image_path

    def estimate_distance(self, object_width_pixels, class_name):
        """Estimate distance based on object width in pixels"""
        if self.FOCAL_LENGTH is None:
            return None
        
        known_width = self.KNOWN_WIDTH.get(class_name, self.DEFAULT_WIDTH)
        distance = (known_width * self.FOCAL_LENGTH) / object_width_pixels
        return distance

    def process_frame(self, frame):
        """Process a frame for face detection/recognition and object detection/distance estimation"""
        start_time = time.time()
        annotated_frame = frame.copy()
        
        # Face Detection and Recognition
        features, faces, aligned_face = self.recognize_face(frame)
        if faces is not None:
            for idx, (face, feature) in enumerate(zip(faces, features)):
                result, user = self.match(feature)
                box = list(map(int, face[:4]))
                color = (0, 255, 0) if result else (0, 0, 255)
                thickness = 2
                cv2.rectangle(annotated_frame, box, color, thickness, cv2.LINE_AA)

                if result:
                    id_name, score = user
                else:
                    id_name = f"unknown_{self.next_unknown_id}"
                    score = 0.0
                    if id_name not in self.detected_unknowns:
                        self.play_notification()
                        self.detected_unknowns.add(id_name)
                        self.save_unknown_face(frame, box, aligned_face, features, self.next_unknown_id)
                        self.dictionary[id_name] = feature
                        self.next_unknown_id += 1

                text = f"{id_name} ({score:.2f})"
                position = (box[0], box[1] - 10)
                cv2.putText(annotated_frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, thickness, cv2.LINE_AA)

        # Object Detection and Distance Estimation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            if not self.model_verbose:
                with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                    results = self.yolo_model(frame_rgb, verbose=False)
            else:
                results = self.yolo_model(frame_rgb)
        except Exception as e:
            print(f"Error during object detection: {e}")
            results = []

        for detection in results:
            if len(detection.boxes) == 0:
                continue
                
            try:
                boxes = detection.boxes.xyxy.cpu().numpy()
                scores = detection.boxes.conf.cpu().numpy()
                class_ids = detection.boxes.cls.cpu().numpy().astype(int)
            except (IndexError, AttributeError):
                continue
            
            for i, box in enumerate(boxes):
                class_id = class_ids[i]
                class_name = self.yolo_model.names[class_id]
                confidence = scores[i]
                
                if confidence < 0.5:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                object_width_pixels = x2 - x1
                
                color = tuple(map(int, self.COLORS[class_id % len(self.COLORS)]))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                distance_text = "Unknown"
                if self.FOCAL_LENGTH is not None:
                    distance = self.estimate_distance(object_width_pixels, class_name)
                    if distance is not None:
                        distance_text = f"{distance:.2f} cm"
                
                label = f"{class_name}: {confidence:.2f}, Dist: {distance_text}"
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y1 = max(y1, label_size[1])
                
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Display FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame

    def toggle_verbose(self):
        """Toggle verbose output mode"""
        self.verbose_mode = not self.verbose_mode
        print(f"Verbose mode {'ON' if self.verbose_mode else 'OFF'}")

    def toggle_model_verbose(self):
        """Toggle model output verbose mode"""
        self.model_verbose = not self.model_verbose
        print(f"Model verbose mode {'ON' if self.model_verbose else 'OFF'}")

    def cleanup(self):
        """Clean up temporary files and directories"""
        if os.path.exists(self.temp_unknown_dir):
            shutil.rmtree(self.temp_unknown_dir)
            print("Cleaned up temporary unknown faces")

    def run(self):
        """Run the combined detection system"""
        try:
            capture = cv2.VideoCapture(0)
            if not capture.isOpened():
                print("Error: Could not open camera")
                return

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 768)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 432)

            print("=== Combined Detection System ===")
            print("Press 'v' to toggle verbose mode")
            print("Press 'm' to toggle model output")
            print("Press 'q' to quit")

            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow("Combined Detection", processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('v'):
                    self.toggle_verbose()
                elif key == ord('m'):
                    self.toggle_model_verbose()

        finally:
            capture.release()
            cv2.destroyAllWindows()
            self.cleanup()

def main():
    detector = CombinedDetector(enable_sound=True)
    detector.run()

if __name__ == '__main__':
    main()