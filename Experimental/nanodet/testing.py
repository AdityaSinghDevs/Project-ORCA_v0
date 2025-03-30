import numpy as np # type: ignore
import cv2 as cv # type: ignore
import argparse
import math

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from nanodet import NanoDet

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

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

# Reference object sizes in cm for distance estimation
# These are approximate average heights for some common objects
reference_heights = {
    'person': 170,      # Average human height in cm
    'car': 150,         # Average car height in cm
    'bicycle': 100,     # Average bicycle height in cm
    'dog': 60,          # Average dog height in cm
    'chair': 80,        # Average chair height in cm
    'bottle': 25        # Average bottle height in cm
}

# Focal length of the camera (needs to be calibrated for your specific camera)
# This is a placeholder value - for accurate results, perform camera calibration
FOCAL_LENGTH = 800

def letterbox(srcimg, target_size=(416, 416)):
    img = srcimg.copy()

    top, left, newh, neww = 0, 0, target_size[0], target_size[1]
    if img.shape[0] != img.shape[1]:
        hw_scale = img.shape[0] / img.shape[1]
        if hw_scale > 1:
            newh, neww = target_size[0], int(target_size[1] / hw_scale)
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            left = int((target_size[1] - neww) * 0.5)
            img = cv.copyMakeBorder(img, 0, 0, left, target_size[1] - neww - left, cv.BORDER_CONSTANT, value=0)  # add border
        else:
            newh, neww = int(target_size[0] * hw_scale), target_size[1]
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            top = int((target_size[0] - newh) * 0.5)
            img = cv.copyMakeBorder(img, top, target_size[0] - newh - top, 0, 0, cv.BORDER_CONSTANT, value=0)
    else:
        img = cv.resize(img, target_size, interpolation=cv.INTER_AREA)

    letterbox_scale = [top, left, newh, neww]
    return img, letterbox_scale

def unletterbox(bbox, original_image_shape, letterbox_scale):
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

def estimate_distance(object_class, bbox_height):
    """
    Estimate distance using the formula:
    distance = (reference_height * focal_length) / apparent_height
    
    Args:
        object_class: Class of the detected object
        bbox_height: Height of the bounding box in pixels
    
    Returns:
        Estimated distance in meters
    """
    # Use default reference height if class not in dictionary
    reference_height = reference_heights.get(object_class, 100)
    
    # Calculate distance in cm, convert to meters
    if bbox_height > 0:
        distance = (reference_height * FOCAL_LENGTH) / bbox_height
        return distance / 100  # Convert cm to meters
    else:
        return None

def vis(preds, res_img, letterbox_scale, fps=None):
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
        class_name = classes[classid]

        # bbox
        xmin, ymin, xmax, ymax = unletterbox(bbox, ret.shape[:2], letterbox_scale)
        cv.rectangle(ret, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)

        # Calculate bbox height
        bbox_height = ymax - ymin
        
        # Estimate distance if class has a reference height
        distance = estimate_distance(class_name, bbox_height)
        
        # label with distance
        if distance is not None:
            label = "{:s}: {:.2f}, {:.1f}m".format(class_name, conf, distance)
        else:
            label = "{:s}: {:.2f}".format(class_name, conf)
            
        cv.putText(ret, label, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), thickness=2)

    return ret

def calibrate_focal_length(known_distance, known_height, bbox_height):
    """
    Calculate focal length using a reference image
    
    Args:
        known_distance: Actual distance (in meters) to the object in reference image
        known_height: Actual height (in cm) of the object
        bbox_height: Height of the object in the image (in pixels)
    
    Returns:
        Focal length
    """
    # Convert distance from meters to cm
    known_distance_cm = known_distance * 100
    return (bbox_height * known_distance_cm) / known_height

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Nanodet inference using OpenCV an contribution by Sri Siddarth Chakaravarthy part of GSOC_2022')
    parser.add_argument('--input', '-i', type=str,
                        help='Path to the input image. Omit for using default camera.')
    parser.add_argument('--model', '-m', type=str,
                        default='models/object_detection_nanodet_2022nov_int8bq.onnx', help="Path to the model")
    parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
    parser.add_argument('--confidence', default=0.35, type=float,
                        help='Class confidence')
    parser.add_argument('--nms', default=0.6, type=float,
                        help='Enter nms IOU threshold')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Specify to save results. This flag is invalid when using camera.')
    parser.add_argument('--vis', '-v', action='store_true',
                        help='Specify to open a window for result visualization. This flag is invalid when using camera.')
    parser.add_argument('--calibrate', '-c', action='store_true',
                        help='Enable camera calibration mode')
    parser.add_argument('--focal_length', type=float, default=800,
                        help='Focal length of camera (default: 800, set after calibration)')
    args = parser.parse_args()

    # Update global focal length if provided
    # global FOCAL_LENGTH 
    FOCAL_LENGTH = args.focal_length

    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    model = NanoDet(modelPath=args.model,
                    prob_threshold=args.confidence,
                    iou_threshold=args.nms,
                    backend_id=backend_id,
                    target_id=target_id)

    tm = cv.TickMeter()
    tm.reset()
    
    # Camera calibration mode
    if args.calibrate and args.input is None:
        print("Calibration mode: Place an object of known height at a known distance")
        print("Press 'c' to capture calibration image, 'q' to quit")
        
        deviceId = 0
        cap = cv.VideoCapture(deviceId)
        
        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break
                
            # Display calibration instructions
            cv.putText(frame, "Calibration Mode", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.putText(frame, "Place object at known distance", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.putText(frame, "Press 'c' to capture, 'q' to quit", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv.imshow("Calibration", frame)
            
            key = cv.waitKey(1) & 0xFF
            if key == ord('c'):
                # Process calibration image
                input_blob = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                input_blob, letterbox_scale = letterbox(input_blob)
                
                # Run detection
                preds = model.infer(input_blob)
                
                if len(preds) > 0:
                    # Get the first detected object
                    bbox = preds[0][:4]
                    classid = int(preds[0][-1])
                    class_name = classes[classid]
                    
                    # Get bounding box coordinates
                    xmin, ymin, xmax, ymax = unletterbox(bbox, frame.shape[:2], letterbox_scale)
                    bbox_height = ymax - ymin
                    
                    # Ask for actual measurements
                    distance = float(input("Enter actual distance to object (meters): "))
                    
                    if class_name in reference_heights:
                        height = reference_heights[class_name]
                        print(f"Using standard height for {class_name}: {height} cm")
                    else:
                        height = float(input("Enter actual height of object (cm): "))
                        
                    # Calculate focal length
                    new_focal_length = calibrate_focal_length(distance, height, bbox_height)
                    print(f"Calculated focal length: {new_focal_length}")
                    print(f"Use this value with --focal_length parameter")
                    
                    # Update global focal length
                    FOCAL_LENGTH = new_focal_length
                else:
                    print("No objects detected in calibration image")
            
            elif key == ord('q'):
                break
                
        cap.release()
        cv.destroyAllWindows()
        
    elif args.input is not None:
        image = cv.imread(args.input)
        input_blob = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Letterbox transformation
        input_blob, letterbox_scale = letterbox(input_blob)

        # Inference
        tm.start()
        preds = model.infer(input_blob)
        tm.stop()
        print("Inference time: {:.2f} ms".format(tm.getTimeMilli()))

        img = vis(preds, image, letterbox_scale)

        if args.save:
            print('Results saved to result.jpg\n')
            cv.imwrite('result.jpg', img)

        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, img)
            cv.waitKey(0)

    else:
        print("Press any key to stop video capture")
        deviceId = 0
        cap = cv.VideoCapture(deviceId)

        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            input_blob = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            input_blob, letterbox_scale = letterbox(input_blob)
            # Inference
            tm.start()
            preds = model.infer(input_blob)
            tm.stop()

            img = vis(preds, frame, letterbox_scale, fps=tm.getFPS())

            cv.imshow("NanoDet Demo", img)

            tm.reset()