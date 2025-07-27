import cv2
import numpy as np
import csv, os, math
import yaml

from Pre_Processing import _load_calibration, _undistort_image, pre_processing, alignment_cropping, axes_of_symmetry, Convex_Hull
from Symmetry_Via_Moments import axial_symmetry_detection_m
from Symmetry_Via_Rectangles import axial_symmetry_detection_r
from Utils import visualization

# ─────────────────── PARAMETERS ─────────────────── #

IMG_PATH   = "immagini/setup_definitivo/IMG_5.bmp"
CALIB_PATH = "calib_data/calib.npz"

# Directory of destination (created if it doesn't exist)
OUT_DIR = "immagini/aligned_cropped"
os.makedirs(OUT_DIR, exist_ok=True)

# File name of the cropped image derived automatically
cropped_name = os.path.splitext(os.path.basename(IMG_PATH))[0] + "_processed.png"
CROPPED_PATH = os.path.join(OUT_DIR, cropped_name)

W = 2048  # sensor width
H = 1088  # sensor height

# ─────────────────── FUNCTIONS ─────────────────── #

def _visualization(img):
    """
    Display an image in a resizable OpenCV window.

    Parameters:
        img (np.ndarray): Image to display (grayscale or BGR).
    """
    w, h = img.shape[:2]
    ratio = w / h 
    if ratio > 1:
        w_disp = 800
        h_disp = int(800 / ratio)
    else:
        h_disp = 800
        w_disp = int(800 * ratio)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", w_disp, h_disp)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


# ─────────────────── MAIN ─────────────────── #

image_m = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
image_r = image_m.copy()
mtx, dist = _load_calibration(CALIB_PATH)    
image_undistorted_m = _undistort_image(image_m, mtx, dist)
image_undistorted_r = _undistort_image(image_r, mtx, dist)

binary_m = pre_processing(image_m)
binary_hull_m = Convex_Hull(binary_m) 
image_m, binary_m = alignment_cropping(binary_hull_m, image_undistorted_m, CROPPED_PATH)
axes_of_symmetry(binary_m, IMG_PATH, k = "m")


binary_r = pre_processing(image_r) 
image_r, binary_r = alignment_cropping(binary_r, image_undistorted_r, CROPPED_PATH) 
axes_of_symmetry(binary_r, IMG_PATH, k = "r") 


img_color_r = axial_symmetry_detection_r(binary_r, IMG_PATH, image_r)
_visualization(img_color_r)


img_color_m, img_color_2 = axial_symmetry_detection_m(binary_m, IMG_PATH, image_m)
_visualization(img_color_m)
 






        











