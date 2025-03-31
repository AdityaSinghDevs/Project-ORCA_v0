import os
import cv2
import numpy as np
import glob
from flask import Flask, render_template, Response
import time
from datetime import datetime
from playsound import playsound
from tqdm import tqdm
import requests
from PIL import Image
import io
import re
import shutil
import atexit

app = Flask(__name__)

# Constants
COSINE_THRESHOLD = 0.5
DIRECTORY = 'data'
ESP32_CAM_URL = "http://192.168.118.200:81/stream"  # Replace with your ESP32-CAM IP address
CHUNK_SIZE = 512  # Reduced chunk size for faster processing
FRAME_INTERVAL = 0.05  # 50ms between frames (20 FPS)

# Initialize face detection and recognition models
weights = os.path.join(DIRECTORY, "models", "face_detection_yunet_2023mar.onnx")
face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
face_detector.setScoreThreshold(0.87)

weights = os.path.join(DIRECTORY, "models", "face_recognition_sface_2021dec_int8bq.onnx")
face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

# Global variables
dictionary = {}
next_unknown_id = 1
detected_unknowns = set()
last_frame_time = 0

def get_esp32_frame():
    global last_frame_time
    current_time = time.time()
    
    # Limit frame rate
    if current_time - last_frame_time < FRAME_INTERVAL:
        time.sleep(0.001)  # Small sleep to prevent CPU overuse
        return False, None
        
    try:
        response = requests.get(ESP32_CAM_URL, stream=True)
        if response.status_code == 200:
            bytes_data = bytes()
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        last_frame_time = current_time
                        return True, frame
    except Exception as e:
        print(f"Error getting frame from ESP32-CAM: {e}")
    return False, None

def match(recognizer, feature1, dictionary):
    max_score = 0.0
    sim_user_id = ""
    for user_id, feature2 in zip(dictionary.keys(), dictionary.values()):
        score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        if score >= max_score:
            max_score = score
            sim_user_id = user_id
    if max_score < COSINE_THRESHOLD:
        return False, ("", 0.0)
    return True, (sim_user_id, max_score)

def recognize_face(image, face_detector, face_recognizer, file_name=None):
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.shape[0] > 1000:
        image = cv2.resize(image, (0, 0), fx=500 / image.shape[0], fy=500 / image.shape[0])

    height, width, _ = image.shape
    face_detector.setInputSize((width, height))
    try:
        _, faces = face_detector.detect(image)
        if file_name is not None:
            assert len(faces) > 0, f'the file {file_name} has no face'

        faces = faces if faces is not None else []
        features = []
        for face in faces:
            aligned_face = face_recognizer.alignCrop(image, face)
            feat = face_recognizer.feature(aligned_face)
            features.append(feat)
        return features, faces, aligned_face
    except Exception as e:
        print(e)
        print(file_name)
        return None, None, None

def save_unknown_face(image, face_box, aligned_face, features, unknown_id):
    unknown_dir = os.path.join(DIRECTORY, 'unknown_faces')
    os.makedirs(unknown_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"unknown_{unknown_id}_{timestamp}.jpg"
    image_path = os.path.join(unknown_dir, image_filename)
    cv2.imwrite(image_path, aligned_face)
    
    feature_filename = f"unknown_{unknown_id}_{timestamp}.npy"
    feature_path = os.path.join(unknown_dir, feature_filename)
    np.save(feature_path, features[0])
    
    return image_path

def load_unknown_faces():
    unknown_dir = os.path.join(DIRECTORY, 'unknown_faces')
    if not os.path.exists(unknown_dir):
        return {}
    
    unknown_dict = {}
    npy_files = glob.glob(os.path.join(unknown_dir, '*.npy'))
    
    for npy_file in npy_files:
        unknown_id = os.path.basename(npy_file).split('_')[1]
        feature = np.load(npy_file)
        unknown_dict[f"unknown_{unknown_id}"] = feature
    
    return unknown_dict

def load_known_faces():
    global dictionary, next_unknown_id
    types = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG')
    files = []
    for a_type in types:
        files.extend(glob.glob(os.path.join(DIRECTORY, 'images', a_type)))

    files = list(set(files))

    for file in tqdm(files):
        image = cv2.imread(file)
        feats, faces, _ = recognize_face(image, face_detector, face_recognizer, file)
        if faces is None:
            continue
        user_id = os.path.splitext(os.path.basename(file))[0]
        dictionary[user_id] = feats[0]

    # Load previously stored unknown faces
    unknown_dict = load_unknown_faces()
    dictionary.update(unknown_dict)
    next_unknown_id = len(unknown_dict) + 1

def generate_frames():
    global next_unknown_id, detected_unknowns
    while True:
        success, frame = get_esp32_frame()
        if not success:
            time.sleep(0.001)  # Reduced sleep time
            continue

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))  # Adjust size as needed

        features, faces, aligned_face = recognize_face(frame, face_detector, face_recognizer)
        if faces is not None:
            for idx, (face, feature) in enumerate(zip(faces, features)):
                result, user = match(face_recognizer, feature, dictionary)
                box = list(map(int, face[:4]))
                color = (0, 255, 0) if result else (0, 0, 255)
                thickness = 2
                cv2.rectangle(frame, box, color, thickness, cv2.LINE_AA)

                if result:
                    id_name, score = user
                else:
                    id_name = f"unknown_{next_unknown_id}"
                    score = 0.0
                    if id_name not in detected_unknowns:
                        playsound("beep.mp3")
                        detected_unknowns.add(id_name)
                        save_unknown_face(frame, box, aligned_face, features, next_unknown_id)
                        dictionary[id_name] = feature
                        next_unknown_id += 1

                text = "{0} ({1:.2f})".format(id_name, score)
                position = (box[0], box[1] - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.6
                cv2.putText(frame, text, position, font, scale, color, thickness, cv2.LINE_AA)

        # Compress the frame for faster transmission
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # Reduced quality for faster transmission
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('stream.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def cleanup_unknown_faces():
    """Clean up unknown faces directory when the application exits"""
    unknown_dir = os.path.join(DIRECTORY, 'unknown_faces')
    if os.path.exists(unknown_dir):
        try:
            shutil.rmtree(unknown_dir)
            print("Cleaned up unknown faces directory")
        except Exception as e:
            print(f"Error cleaning up unknown faces: {e}")

# Register the cleanup function to run on exit
atexit.register(cleanup_unknown_faces)

if __name__ == '__main__':
    load_known_faces()
    app.run(host='0.0.0.0', port=5000, debug=True) 