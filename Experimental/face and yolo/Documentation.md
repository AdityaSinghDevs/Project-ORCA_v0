This model runs YuNet for face detection, SFace for face recognition and Yolov10 for object detection.
Face images are to be stored in data/images with image named by the name of person.

unknown faces are stored temporarily and then deleted after terminating the model.
script.py runs all the models parallaly.
face.py runs the model SFace and YuNet parallelly which recognizes known and unknown faces.
To run only yolov10 run yolov10_inference.py

esp32_processor.py runs ths model on esp32 when connected to hotspot, changes might be neede with the ip address of video stream depending on the hotspot and other factors.

updated_script.py runs the models on local cam