import os
import sys
import glob
import time
import math
import cv2
import numpy as np
from tqdm import tqdm
from playsound import playsound
from datetime import datetime

COSINE_THRESHOLD = 0.5

def match(recognizer, feature1, dictionary):
    max_score = 0.0
    sim_user_id = ""
    for user_id, feature2 in zip(dictionary.keys(), dictionary.values()):
        score = recognizer.match(
            feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
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
        image = cv2.resize(image, (0, 0),
                           fx=500 / image.shape[0], fy=500 / image.shape[0])

    height, width, _ = image.shape
    face_detector.setInputSize((width, height))
    try:
        dts = time.time()
        _, faces = face_detector.detect(image)
        if file_name is not None:
            assert len(faces) > 0, f'the file {file_name} has no face'

        faces = faces if faces is not None else []
        features = []
        # print(f'time detection  = {time.time() - dts}')
        for face in faces:
            rts = time.time()
            aligned_face = face_recognizer.alignCrop(image, face)
            feat = face_recognizer.feature(aligned_face)
            # print(f'time recognition  = {time.time() - rts}')
            features.append(feat)
        return features, faces, aligned_face
    except Exception as e:
        print(e)
        print(file_name)
        return None, None, None

def save_unknown_face(image, face_box, aligned_face, features, unknown_id):
    # Create directory for unknown faces if it doesn't exist
    unknown_dir = os.path.join('data', 'unknown_faces')
    os.makedirs(unknown_dir, exist_ok=True)
    
    # Save the aligned face image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"unknown_{unknown_id}_{timestamp}.jpg"
    image_path = os.path.join(unknown_dir, image_filename)
    cv2.imwrite(image_path, aligned_face)
    
    # Save the feature vector
    feature_filename = f"unknown_{unknown_id}_{timestamp}.npy"
    feature_path = os.path.join(unknown_dir, feature_filename)
    np.save(feature_path, features[0])
    
    return image_path

def load_unknown_faces():
    unknown_dir = os.path.join('data', 'unknown_faces')
    if not os.path.exists(unknown_dir):
        return {}
    
    unknown_dict = {}
    npy_files = glob.glob(os.path.join(unknown_dir, '*.npy'))
    
    for npy_file in npy_files:
        unknown_id = os.path.basename(npy_file).split('_')[1]
        feature = np.load(npy_file)
        unknown_dict[f"unknown_{unknown_id}"] = feature
    
    return unknown_dict

def main():
    directory = 'data'

    # Init models face detection & recognition
    weights = os.path.join(directory, "models",
                           "face_detection_yunet_2023mar.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
    face_detector.setScoreThreshold(0.87)

    weights = os.path.join(directory, "models", "face_recognition_sface_2021dec_int8bq.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    # Load registered faces
    dictionary = {}
    types = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG')
    files = []
    for a_type in types:
        files.extend(glob.glob(os.path.join(directory, 'images', a_type)))

    files = list(set(files))

    for file in tqdm(files):
        image = cv2.imread(file)
        feats, faces, _ = recognize_face(
            image, face_detector, face_recognizer, file)
        if faces is None:
            continue
        user_id = os.path.splitext(os.path.basename(file))[0]
        dictionary[user_id] = feats[0]

    # Load previously stored unknown faces
    unknown_dict = load_unknown_faces()
    dictionary.update(unknown_dict)
    
    # print(f'there are {len(dictionary)} ids (including unknowns)')
    
    # Keep track of the next available unknown ID
    next_unknown_id = len(unknown_dict) + 1
    detected_unknowns = set()

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        sys.exit()

    while True:
        start_hand = time.time()
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        features, faces, aligned_face = recognize_face(image, face_detector, face_recognizer)
        if faces is None:
            continue

        for idx, (face, feature) in enumerate(zip(faces, features)):
            result, user = match(face_recognizer, feature, dictionary)
            box = list(map(int, face[:4]))
            color = (0, 255, 0) if result else (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            if result:
                id_name, score = user
            else:
                id_name = f"unknown_{next_unknown_id}"
                score = 0.0
                if id_name not in detected_unknowns:
                    playsound("beep.mp3")
                    detected_unknowns.add(id_name)
                    # Save the unknown face and its features
                    save_unknown_face(image, box, aligned_face, features, next_unknown_id)
                    # Add to dictionary for future recognition
                    dictionary[id_name] = feature
                    next_unknown_id += 1

            text = "{0} ({1:.2f})".format(id_name, score)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            cv2.putText(image, text, position, font, scale,
                        color, thickness, cv2.LINE_AA)

        cv2.imshow("face recognition", image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        end_hand = time.time()
        # print(f'speed of a loop = {end_hand - start_hand} means {1/(end_hand - start_hand)} frames per second')

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()