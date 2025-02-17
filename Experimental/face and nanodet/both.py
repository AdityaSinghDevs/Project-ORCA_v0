import os
import sys
import glob
import time
import cv2 as cv
import numpy as np
from tqdm import tqdm
from nanodet import NanoDet
import argparse

COSINE_THRESHOLD = 0.5


classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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

class CombinedDetector:
    def __init__(self, nanodet_model, confidence=0.35, nms=0.6, 
                 backend_id=cv.dnn.DNN_BACKEND_OPENCV, 
                 target_id=cv.dnn.DNN_TARGET_CPU):
        # Initialize NanoDet
        self.object_detector = NanoDet(
            modelPath=nanodet_model,
            prob_threshold=confidence,
            iou_threshold=nms,
            backend_id=backend_id,
            target_id=target_id
        )
        
        # Initialize Face Detection/Recognition
        directory = 'data'
        face_detect_weights = os.path.join(directory, "models", "face_detection_yunet_2023mar.onnx")
        face_recog_weights = os.path.join(directory, "models", "face_recognition_sface_2021dec_int8bq.onnx")
        
        self.face_detector = cv.FaceDetectorYN_create(face_detect_weights, "", (0, 0))
        self.face_detector.setScoreThreshold(0.87)
        self.face_recognizer = cv.FaceRecognizerSF_create(face_recog_weights, "")
        
        # Load face dictionary
        self.face_dictionary = self.load_face_dictionary(directory)
        
    def load_face_dictionary(self, directory):
        dictionary = {}
        types = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG')
        files = []
        for a_type in types:
            files.extend(glob.glob(os.path.join(directory, 'images', a_type)))
        
        files = list(set(files))
        print("Loading face database...")
        for file in tqdm(files):
            image = cv.imread(file)
            feats, faces = self.recognize_face(image, file_name=file)
            if faces is None:
                continue
            user_id = os.path.splitext(os.path.basename(file))[0]
            dictionary[user_id] = feats[0]
        
        print(f'Loaded {len(dictionary)} face IDs')
        return dictionary
    
    def recognize_face(self, image, file_name=None):
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)

        if image.shape[0] > 1000:
            image = cv.resize(image, (0, 0),
                            fx=500 / image.shape[0], 
                            fy=500 / image.shape[0])

        height, width, _ = image.shape
        self.face_detector.setInputSize((width, height))
        
        try:
            _, faces = self.face_detector.detect(image)
            if file_name is not None:
                assert len(faces) > 0, f'the file {file_name} has no face'

            faces = faces if faces is not None else []
            features = []
            
            for face in faces:
                aligned_face = self.face_recognizer.alignCrop(image, face)
                feat = self.face_recognizer.feature(aligned_face)
                features.append(feat)
                
            return features, faces
        except Exception as e:
            print(e)
            if file_name:
                print(file_name)
            return None, None

    def match_face(self, feature1):
        max_score = 0.0
        sim_user_id = ""
        for user_id, feature2 in self.face_dictionary.items():
            score = self.face_recognizer.match(
                feature1, feature2, cv.FaceRecognizerSF_FR_COSINE)
            if score >= max_score:
                max_score = score
                sim_user_id = user_id
        if max_score < COSINE_THRESHOLD:
            return False, ("", 0.0)
        return True, (sim_user_id, max_score)
    
    def letterbox(self, srcimg, target_size=(416, 416)):
        # Debug print to check input image
        print("Input image shape:", srcimg.shape)
        
        # Check if image is valid
        if srcimg is None or len(srcimg.shape) < 2:
            print("Error: Invalid input image")
            return None, None
            
        img = np.copy(srcimg)
        
        # For grayscale images, convert to BGR
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        
        # For RGBA images, convert to BGR
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

        height, width = img.shape[:2]
        top, left, newh, neww = 0, 0, target_size[0], target_size[1]

        if height != width:
            hw_scale = height / width
            if hw_scale > 1:
                newh, neww = target_size[0], int(target_size[1] / hw_scale)
                img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
                left = int((target_size[1] - neww) * 0.5)
                img = cv.copyMakeBorder(img, 0, 0, left, target_size[1] - neww - left, cv.BORDER_CONSTANT, value=0)
            else:
                newh, neww = int(target_size[0] * hw_scale), target_size[1]
                img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
                top = int((target_size[0] - newh) * 0.5)
                img = cv.copyMakeBorder(img, top, target_size[0] - newh - top, 0, 0, cv.BORDER_CONSTANT, value=0)
        else:
            img = cv.resize(img, target_size, interpolation=cv.INTER_AREA)

        letterbox_scale = [top, left, newh, neww]
        return img, letterbox_scale
    
    def unletterbox(self, bbox, original_image_shape, letterbox_scale):
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
    
    def vis(self, preds, res_img, letterbox_scale, fps=None):
        ret = res_img.copy()

        # draw FPS
        if fps is not None:
            fps_label = "FPS: %.2f" % fps
            cv.putText(ret, fps_label, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # draw bboxes and labels
        for pred in preds:
            bbox = pred[:4]
            conf = pred[-2]
            classid = pred[-1].astype(np.int32)

            # bbox
            xmin, ymin, xmax, ymax = self.unletterbox(bbox, ret.shape[:2], letterbox_scale)
            cv.rectangle(ret, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)

            # label
            label = "{:s}: {:.2f}".format(classes[classid], conf)
            cv.putText(ret, label, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

        return ret

    def process_frame(self, frame):
        # Object Detection
        if frame is None:
            print("Error: Empty frame received")
            return frame

        input_blob = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Add error checking for letterbox
        input_blob_resized, letterbox_scale = self.letterbox(input_blob)
        if input_blob_resized is None:
            print("Error: Letterbox transformation failed")
            return frame
            
        object_preds = self.object_detector.infer(input_blob_resized)
        
        # Draw object detection results
        frame = self.vis(object_preds, frame, letterbox_scale)
        
        # Face Detection and Recognition
        features, faces = self.recognize_face(frame)
        if faces is not None:
            for idx, (face, feature) in enumerate(zip(faces, features)):
                result, user = self.match_face(feature)
                box = list(map(int, face[:4]))
                color = (0, 255, 0) if result else (0, 0, 255)
                thickness = 2
                cv.rectangle(frame, box, color, thickness, cv.LINE_AA)

                id_name, score = user if result else (f"unknown_{idx}", 0.0)
                text = "{0} ({1:.2f})".format(id_name, score)
                position = (box[0], box[1] - 10)
                font = cv.FONT_HERSHEY_SIMPLEX
                scale = 0.6
                cv.putText(frame, text, position, font, scale,
                        color, thickness, cv.LINE_AA)
        
        return frame

def main():
    parser = argparse.ArgumentParser(description='Combined NanoDet and Face Recognition Demo')
    parser.add_argument('--model', '-m', type=str,
                    default='data/models/object_detection_nanodet_2022nov_int8bq.onnx',
                    help="Path to the NanoDet model")
    parser.add_argument('--confidence', default=0.35, type=float,
                    help='Class confidence')
    parser.add_argument('--nms', default=0.6, type=float,
                    help='NMS IOU threshold')
    args = parser.parse_args()

    # Initialize combined detector
    detector = CombinedDetector(
        nanodet_model=args.model,
        confidence=args.confidence,
        nms=args.nms
    )

    # Start video capture
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Could not open camera")
        sys.exit()

    print("Press 'q' to quit")
    while True:
        start_time = time.time()
        ret, frame = capture.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Process frame with both detectors
        processed_frame = detector.process_frame(frame)

        # Calculate and display FPS
        fps = 1.0 / (time.time() - start_time)
        cv.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the result
        cv.imshow("Combined Detection", processed_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()