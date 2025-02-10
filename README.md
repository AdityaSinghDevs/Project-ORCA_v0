
# Project ORCAv0 (Omni-Purpose Real-time Computer-vision Aid)
<h4>Project repo for ORCA</h4>

This project integrates a camera module with an edge device to build a pair of wearable glasses that provide real-time object detection, translation, and gesture controls. The device uses a transparent OLED screen to display detected objects, translations, and other outputs to assist users, particularly those who are blind or visually impaired.

## Features

### Planned MVP Features:
1. **Real-Time Object Detection/Classification**:  
   Detect and classify objects in the field of view using the YOLO model. The system also performs **Monocular Depth Estimation** to assist with navigation for blind individuals.
   
2. **Real-Time Translation**:  
   Translate text from the field of view in real-time and display it on the OLED screen.

3. **Gesture Controls**:  
   Use predefined hand gestures to activate specific features (like starting object detection or translation).

### Future Features:
4. **LLM Support**:  
   AI-based assistance using large language models.

5. **Navigation Maps**:  
   Display navigation directions on the transparent OLED screen.

6. **Facial Detection**:  
   Detect facial attributes like gender, age, and expressions for enhanced interaction.

7. **SOS Alerts based on visuals**
     Give off SOS signal based on the visual patterns it sees

---

```
ORCA/
├── data/                                # All datasets and data loaders
│     ├── __init__.py                    # Initialization for the data package
│     ├── coco_loader.py                 # Script to load and process COCO dataset
│     ├── data.md                        # Documentation for the data folder
│     └── ...
│
├── Experimental/                        # Experimental models and code 
│     ├── face recognition/              # Contains sub-directories for experimenting on features
│     │     ├── face_classifier.pkl      # Pre-trained face classification model
│     │     ├── face_recognition.py      # Code for face recognition tasks
│     │     ├── label_encoder.pkl        # Encoded labels for classification
│     │     └── ...
│     ├── object detection/              # sub-directory for object detection
│     └── ...                            # Experimental feature directories
│
├── notebooks/                           # Jupyter notebooks for prototyping and exploration
│     └── ...  
│
├── src/                                 # Source code for the main application
│     ├── inference/                     # Real-time deployment files
│     │     ├── __init__.py              # Initialization for the inference package 
│     │     ├── detection.py             # Core logic for feature detection and inference in real-time
│     │     └── deployment.py            # Real-time deployment pipeline
│     │
│     ├── models/                        # Training model scripts
│     │     ├── model1/                 
│     │     │     ├── __init__.py        # Initialization for the model 1 package 
│     │     │     ├── training.py        # Scripts to train the model 1
│     │     │     └── utils.py           # Utility functions for model 1
│     │     ├── model2/                  # Scripts and utilities specific to Model 2
│     │     └── ...                      # Additional model sub-directories
│     │
│     ├── trained_model/                 # All trained models are stored in this directory
│     │     ├── model_v1.pth             # Trained model 1 
│     │     ├── model_v2.pth             # Trained model 2
│     │     └── ...
│     │
│     ├── __init__.py                    # Initialization for the src package
│     ├── README.md                      # Documentation for the src folder
│     └── utils.py                       # Contains helper functions
│
├── test/                                # Unit and integration test cases
│     ├── test_inference.py              # Tests for inference components 
│     ├── test_training.py               # Tests for training components
│     └── test.md                        # Documentation or description for tests
│
├── main.py                              # Entry point to run the entire program
├── README.md                            # Project overview and instructions
└── requirements.txt                     # Dependencies for the project
```


<script "https://gist.github.com/Nivratti/ea81e952e07ffbbf03e6d44a7dbbef8f.js"
