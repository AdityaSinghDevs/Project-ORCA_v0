'''
Supposed to load Coco (Common Object in Context) using Kaggle API key

Summary of the Process:
Load COCO Annotations:

Load the COCO annotations (JSON format) and initialize the pycocotools.COCO API for easier processing.
Get Image Details:

For each image in the dataset, retrieve its filename and its bounding box annotations.
Convert Bounding Boxes:

Convert the COCO bounding box format into YOLO format (normalized center and dimensions).
Write YOLO Annotations:

For each image, create a .txt file with YOLO-formatted annotations (one text file per image).
'''