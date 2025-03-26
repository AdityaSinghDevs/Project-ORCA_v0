import os
import sys
import glob
import time
import math
import cv2
import numpy as np
import shutil
import platform
from tqdm import tqdm
from datetime import datetime
from nanodet import NanoDet

class CombinedDetector:
    def __init__(self, enable_sound=True):
        self.directory = 'data'
        self.COSINE_THRESHOLD = 0.5
        self.temp_unknown_dir = os.path.join(self.directory, 'temp_unknown_faces')
        self.enable_sound = enable_sound
        
        # Setup sound based on platform
        if self.enable_sound:
            self.setup_sound()

        self.setup_face_detection()
        self.setup_object_detection()
        self.setup_temp_directory()
        self.load_face_dictionary()
        self.next_unknown_id = 1
        self.detected_unknowns = set()

    def setup_sound(self):
        """Setup sound based on platform"""
        self.system = platform.system()
        if self.system == 'Windows':
            try:
                import winsound
                self.sound_function = lambda: winsound.Beep(1000, 700)  # 1000Hz for 700ms
                self.sound_available = True
            except Exception as e:
                print(f"Warning: Could not initialize Windows sound: {e}")
                self.sound_available = False
        elif self.system == 'Darwin':  # macOS
            self.sound_function = lambda: os.system('afplay /System/Library/Sounds/Ping.aiff')
            self.sound_available = True
        elif self.system == 'Linux':
            # Try to use console bell
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
        if os.path.exists(self.temp_unknown_dir):
            shutil.rmtree(self.temp_unknown_dir)
        os.makedirs(self.temp_unknown_dir)

    def setup_face_detection(self):
        weights = os.path.join(self.directory, "models", "face_detection_yunet_2023mar.onnx")
        self.face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
        self.face_detector.setScoreThreshold(0.87)

        weights = os.path.join(self.directory, "models", "face_recognition_sface_2021dec_int8bq.onnx")
        self.face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    def setup_object_detection(self):
        # Replace YOLOv10 with NanoDetect
        model_path = os.path.join(self.directory, "models", "object_detection_nanodet_2022nov_int8bq.onnx")
        self.nanodet_model = NanoDet(modelPath=model_path, 
                                       prob_threshold=0.35, 
                                       iou_threshold=0.6, 
                                       backend_id=cv2.dnn.DNN_BACKEND_OPENCV, 
                                       target_id=cv2.dnn.DNN_TARGET_CPU)
        
        # Define class names for NanoDetect
        self.class_names = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                  'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                  'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                  'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                  'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                  'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                  'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                  'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                  'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                  'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                  'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    def load_face_dictionary(self):
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

    def match(self, feature1):
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"unknown_{unknown_id}_{timestamp}.jpg"
        image_path = os.path.join(self.temp_unknown_dir, image_filename)
        cv2.imwrite(image_path, aligned_face)
        
        feature_filename = f"unknown_{unknown_id}_{timestamp}.npy"
        feature_path = os.path.join(self.temp_unknown_dir, feature_filename)
        np.save(feature_path, features[0])
        
        return image_path

    def letterbox(self, srcimg, target_size=(416, 416)):
        # Add the letterbox function from NanoDetect demo
        img = srcimg.copy()

        top, left, newh, neww = 0, 0, target_size[0], target_size[1]
        if img.shape[0] != img.shape[1]:
            hw_scale = img.shape[0] / img.shape[1]
            if hw_scale > 1:
                newh, neww = target_size[0], int(target_size[1] / hw_scale)
                img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((target_size[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, target_size[1] - neww - left, cv2.BORDER_CONSTANT, value=0)
            else:
                newh, neww = int(target_size[0] * hw_scale), target_size[1]
                img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((target_size[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, target_size[0] - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        letterbox_scale = [top, left, newh, neww]
        return img, letterbox_scale

    def unletterbox(self, bbox, original_image_shape, letterbox_scale):
        # Add the unletterbox function from NanoDetect demo
        ret = bbox.copy()

        h, w = original_image_shape
        top, left, newh, neww = letterbox_scale

        if h == w:
            ratio = h / newh
            ret = ret * ratio
            return ret

        ratioh, ratiow = h / newh, w / neww
        ret[0] = max((ret[0] - left) * ratiow, 0)
        ret[1] = max((ret[1] - top) * ratioh, 0)
        ret[2] = min((ret[2] - left) * ratiow, w)
        ret[3] = min((ret[3] - top) * ratioh, h)

        return ret.astype(np.int32)

    def process_frame(self, frame):
        start_time = time.time()
        
        # Face Detection and Recognition
        features, faces, aligned_face = self.recognize_face(frame)
        if faces is not None:
            for idx, (face, feature) in enumerate(zip(faces, features)):
                result, user = self.match(feature)
                box = list(map(int, face[:4]))
                color = (0, 255, 0) if result else (0, 0, 255)
                thickness = 2
                cv2.rectangle(frame, box, color, thickness, cv2.LINE_AA)

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
                cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, thickness, cv2.LINE_AA)

        # Object Detection with NanoDetect
        # Convert to RGB for NanoDetect
        input_blob = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply letterbox transformation
        input_blob, letterbox_scale = self.letterbox(input_blob)
        
        # Run inference
        preds = self.nanodet_model.infer(input_blob)
        
        # Draw bounding boxes for detected objects
        if len(preds) > 0:
            for pred in preds:
                bbox = pred[:4]
                conf = pred[-2]
                class_id = int(pred[-1])
                
                # Convert bbox from letterbox coordinates to original image coordinates
                xmin, ymin, xmax, ymax = self.unletterbox(bbox, frame.shape[:2], letterbox_scale)
                
                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                
                # Draw label
                label = f"{self.class_names[class_id]}: {conf:.2f}"
                cv2.putText(frame, label, (xmin, ymin - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

    def cleanup(self):
        """Clean up temporary files and directories"""
        if os.path.exists(self.temp_unknown_dir):
            shutil.rmtree(self.temp_unknown_dir)
            print("Cleaned up temporary unknown faces")

    def run(self):
        try:
            capture = cv2.VideoCapture(0)
            if not capture.isOpened():
                print("Error: Could not open camera")
                return

            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow("Combined Detection", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            capture.release()
            cv2.destroyAllWindows()
            self.cleanup()

def main():
    # Initialize with sound enabled by default
    detector = CombinedDetector(enable_sound=True)
    detector.run()

if __name__ == '__main__':
    main()
