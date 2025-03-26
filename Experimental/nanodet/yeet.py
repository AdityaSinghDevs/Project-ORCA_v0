import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Prepare object points (3D points in real world space)
checkerboard_size = (9, 6)  # Adjust this based on your checkerboard
square_size = 1  # Size of one square in your checkerboard (in any unit, e.g., cm)

obj_points = []  # 3D points in world space
img_points = []  # 2D points in image plane

# Prepare 3D points for the checkerboard
objp = np.zeros((np.prod(checkerboard_size), 3), dtype=np.float32)
objp[:, :2] = np.indices(checkerboard_size).T.reshape(-1, 2)
objp *= square_size

# Collect images
images = glob.glob('calibration_images/*.jpg')  # Replace with your folder path

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        img_points.append(corners)
        obj_points.append(objp)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
        
        # Use matplotlib to display the image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.waitKey(500)

# Ensure we have valid points before calibrating
if len(obj_points) > 0 and len(img_points) > 0:
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    
    # Focal length is stored in the camera matrix (mtx)
    focal_length = mtx[0, 0]  # Typically the focal length is in the [0, 0] element of the matrix
    print("Focal length: ", focal_length)
else:
    print("No valid calibration points found.")
