import os
import sys
import time
import cv2
import numpy as np
import pandas as pd

COSINE_THRESHOLD = 0.5

def normalize_feature(feature):
    return feature / np.linalg.norm(feature)



def match(recognizer, feature1, dictionary):
    max_score = 0.0
    sim_user_id = ""
    feature1 = normalize_feature(feature1)

    for user_id, embedding in dictionary.items():
        print(f"User ID: {user_id}, Embedding Shape: {embedding.shape}")

    for user_id, feature2 in dictionary.items():
        dictionary[user_id] = normalize_feature(feature2)
        try:
            score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
            if score >= max_score:
                max_score = score
                sim_user_id = user_id
        except cv2.error as e:
            print(f"Error matching with {user_id}: {e}")
            continue
    
    if max_score < COSINE_THRESHOLD:
        return False, ("", 0.0)
    return True, (sim_user_id, max_score)

def recognize_face(image, face_detector, face_recognizer):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    height, width, _ = image.shape
    face_detector.setInputSize((width, height))
    
    try:
        _, faces = face_detector.detect(image)
        faces = faces if faces is not None else []
        features = []
        
        for face in faces:
            aligned_face = face_recognizer.alignCrop(image, face)
            feat = face_recognizer.feature(aligned_face)
            features.append(feat)
        
        return features, faces
    except Exception as e:
        print(f"Error during face recognition: {e}")
        return None, None

def load_embeddings(embeddings_file):
    dictionary = {}
    file_extension = os.path.splitext(embeddings_file)[-1].lower()
    
    if file_extension == ".npy":
        data = np.load(embeddings_file, allow_pickle=True).item()
        dictionary.update(data)
    elif file_extension == ".csv":
        
        df = pd.read_csv(embeddings_file)
        for _, row in df.iterrows():
            user_id = row['Person']
            embedding_str = row['Embedding'].strip('[]').split(',')
            embedding = np.array([float(x) for x in embedding_str], dtype=np.float32)
            dictionary[user_id] = embedding
    else:
        raise ValueError("Unsupported embeddings file format. Use .npy or .csv.")
    
    print(f"Loaded {len(dictionary)} embeddings.")
    return dictionary

def main():
    directory = 'data'
    embeddings_file = "face_embeddings.csv"
    dictionary = load_embeddings(embeddings_file)

    weights_detect = os.path.join(directory, "models", "face_detection_yunet_2023mar.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights_detect, "", (0, 0))
    face_detector.setScoreThreshold(0.87)

    weights_recog = os.path.join(directory, "models", "face_recognition_sface_2021dec_int8bq.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights_recog, "")

    print(f'There are {len(dictionary)} registered IDs.')

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        sys.exit("Error: Could not open webcam.")

    while True:
        result, image = capture.read()
        if not result:
            break

        features, faces = recognize_face(image, face_detector, face_recognizer)
        if faces is None or features is None:
            continue

        for idx, (face, feature) in enumerate(zip(faces, features)):
            print(f"Feature {idx} shape: {feature.shape}")
            result, user_info = match(face_recognizer, feature, dictionary)
            box = list(map(int, face[:4]))
            color = (0, 255, 0) if result else (0, 0, 255)
            cv2.rectangle(image, box, color, 2, cv2.LINE_AA)

            id_name, score = user_info if result else (f"unknown_{idx}", 0.0)
            text = f"{id_name} ({score:.2f})"
            cv2.putText(image, text, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Face Recognition", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
