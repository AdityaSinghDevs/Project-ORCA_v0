This model runs YuNet for face detection, SFace for face recognition and NanoDet for object detection.
Face images are to be stored in data/images with image named by the name of person.

unknown faces are stored temporarily and then deleted after terminating the model.
script.py runs all the models parallaly.
face.py runs the model SFace and YuNet parallelly which recognizes known and unknown faces.
To run only nanodet run nanodet.py

