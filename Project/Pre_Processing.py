import cv2
import numpy as np
import csv, os, math
import yaml

from Utils import (
    load_config_yaml,
    save_axes,
    visualization
)


# ─────────────────── PARAMETERS ─────────────────── #

IMG_PATH   = "immagini/setup_definitivo/IMG_17.bmp"
CSV_PATH   = "axes_output.csv"
CALIB_PATH = "calib_data/calib.npz"

# Directory of destination (created if it doesn't exist)
OUT_DIR = "immagini/aligned_cropped"
os.makedirs(OUT_DIR, exist_ok=True)

# File name of the cropped image derived automatically
cropped_name = os.path.splitext(os.path.basename(IMG_PATH))[0] + "_processed.png"
cropped_path = os.path.join(OUT_DIR, cropped_name)

W = 2048  # sensor width
H = 1088  # sensor height
ratio = W / H

# ─────────────────── INTERNAL FUNCTIONS ─────────────────── #


def _load_config_yaml(filepath):
    """
    Load YAML configuration file into a Python dictionary.

    Parameters:
        filepath (str): Path to the YAML file.

    Returns:
        dict: Parsed configuration parameters.
    """
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config

def _endpoints(center, v, img_shape):
    """
    Compute two endpoints along a given direction vector `v`, starting from the center,
    and extending to the image boundaries.

    Parameters:
        center (tuple): (x, y) coordinates of the origin point.
        v (np.ndarray): 2D direction vector.
        img_shape (tuple): (height, width) of the image.

    Returns:
        tuple: Two endpoints (p0, p1) on the line.
    """
    h, w = img_shape
    cx, cy = center
    ts = []
    if v[0]:
        ts += [(-cx)/v[0], (w-1-cx)/v[0]]
    if v[1]:
        ts += [(-cy)/v[1], (h-1-cy)/v[1]]
    p0 = (int(cx + min(ts)*v[0]), int(cy + min(ts)*v[1]))
    p1 = (int(cx + max(ts)*v[0]), int(cy + max(ts)*v[1]))
    return p0, p1
    
def _external_contour(bin_img):
    """
    Find the largest external contour in a binary image and compute its oriented bounding box.

    Parameters:
        bin_img (np.ndarray): Binary image.

    Returns:
        tuple: Center (cx, cy), size (w, h), and rotation angle (theta) in degrees.
    """
    cnt = max(cv2.findContours(bin_img, cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_NONE)[0],
                           key=cv2.contourArea)

    (cx, cy), (w0, h0), theta0 = cv2.minAreaRect(cnt)

    # Angle normalitation: I want the angle in [-45°, 45°] with w = "width of the segment which defines theta0"
    if theta0 > 45:
        w0, h0 = h0, w0
        theta0 -= 90.0   
    return (cx, cy), (w0, h0), theta0

def _load_calibration(calib_file_path):
    """
    Load camera calibration data saved with numpy savez.
    
    Parameters:
        calib_file_path (str): Path to the .npz file containing calibration data.
        
    Returns:
        camera_matrix (np.ndarray): Intrinsic camera matrix.
        dist_coeffs (np.ndarray): Distortion coefficients.
    """
    data = np.load(calib_file_path)
    camera_matrix = data["camMatrix"]
    dist_coeffs = data["distCoef"]
    return camera_matrix, dist_coeffs


def _undistort_image(img, camera_matrix, dist_coeffs, alpha=0.0):
    """
    Undistort an image using precomputed calibration parameters.
    
    Parameters:
        img (np.ndarray): Input distorted image.
        camera_matrix (np.ndarray): Intrinsic camera matrix.
        dist_coeffs (np.ndarray): Distortion coefficients.
        alpha (float): Free scaling parameter between 0 (crop all black pixels) and 1 (retain all pixels).
        
    Returns:
        dst_cropped (np.ndarray): Undistorted image.
    """
    # Image dimensions
    h, w = img.shape[:2]
    
    # Compute optimal new camera matrix
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), alpha, (w, h))
    
    # Undistort
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_mtx)
    
    # Crop the image to the valid ROI
    x, y, cw, ch = roi 
    return dst[y:y+ch, x:x+cw]

# ─────────────────── PRE-PROCESSING ─────────────────── #

def pre_processing(gray_img, canny_params_path='config.yaml', calib_path="calib_data/calib.npz"):
    """
    Preprocess the input image by undistorting it, applying a Gaussian blur, then detecting edges using the Canny operator 
    (with thresholds from a YAML file), followed by dilation and morphological closing.

    Parameters:
        canny_params_path (str): Path to the YAML file containing Canny parameters.
        gray_img (np.ndarray): Gray scale image.
        calib_path (str): Path to the camera calibration file (.npz).
        
    Returns:
        np.ndarray: Binary image of preprocessed edges.

    """
    config = load_config_yaml(canny_params_path)

    blur_kernel = config['blur_kernel']
    sigma = config['sigma']
    threshold1 = config['threshold1']
    threshold2 = config['threshold2']

    mtx, dist = _load_calibration(calib_path)
    
    gray_undistorted = _undistort_image(gray_img, mtx, dist)

    blurred = cv2.GaussianBlur(gray_undistorted, (blur_kernel, blur_kernel), sigmaX=sigma, sigmaY=sigma)

    # Finding of a binary image with canny detected edges as white pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_canny   = cv2.Canny(blurred, threshold1, threshold2) 
    binary_canny   = cv2.dilate(binary_canny, kernel, iterations=1)
    return cv2.morphologyEx(binary_canny, cv2.MORPH_CLOSE, kernel, iterations=1)

# ─────────────── 3. ALIGNMENT & CROPPING ──────────── #

def alignment_cropping(bin_img, gray_img, cropped_path):
    """
    Rotate and crop the image based on the orientation of the largest external contour.

    Parameters:
        bin_img (np.ndarray): Binary image (e.g., edges).
        gray_img (np.ndarray): Corresponding grayscale image.
        cropped_path (str): Path where to save the cropped result.

    Returns:
        tuple: (gray_roi, bin_roi), aligned and cropped grayscale and binary images.
    """
    (cx, cy), (w0, h0), theta0 = _external_contour(bin_img)

    # Rotation matrix
    M = cv2.getRotationMatrix2D((cx, cy), theta0, 1.0)  # clock-wise rotation
    h, w = bin_img.shape
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy

    # Rotation of the original undistorted image and binary_Canny_edges
    gray_rot   = cv2.warpAffine(gray_img, M, (new_w, new_h), flags=cv2.INTER_NEAREST)  
    bin_rot    = cv2.warpAffine(bin_img, M, (new_w, new_h), flags=cv2.INTER_NEAREST)  

    # External contour is found on the aligned image
    cnt_rot = max(cv2.findContours(bin_rot, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)[0],
                key=cv2.contourArea)

    x, y, w_box, h_box = cv2.boundingRect(cnt_rot)

    # Save the cropped image for debugging
    cv2.imwrite(cropped_path, gray_rot[y:y+h_box, x:x+w_box]) 

    return gray_rot[y:y+h_box, x:x+w_box], bin_rot[y:y+h_box, x:x+w_box]    
         


#  ─────────────── 4. AXES OF SYMMETRY  ─────────────── #

def axes_of_symmetry(bin_roi, img_path, k):
    """
    Compute symmetry axes from the binary ROI.

    The function finds the principal orientation using the external contour, computes
    horizontal and vertical symmetry axes, saves their center in a CSV file, and visualizes them.

    Parameters:
        bin_roi (np.ndarray): Binary region of interest.
        img_path (str): Path to the original image (used for logging and CSV).
        csv_path (str): Path to the CSV file for axis logging (default = "axes_output.csv").
    """
    (cx_r, cy_r), (w_r, h_r), theta_r = _external_contour(bin_roi);
            
    # Unitary axes vectors
    vx = np.array([ math.cos(math.radians(theta_r)),
                    math.sin(math.radians(theta_r))])
    vy = np.array([-vx[1], vx[0]])

    p0, p1 = _endpoints((cx_r, cy_r), vx, bin_roi.shape)   # asse “verticale”
    q0, q1 = _endpoints((cx_r, cy_r), vy, bin_roi.shape)   # asse “orizzontale”

    # Update the CSV file with the axes data
    row = dict(image=os.path.basename(img_path),
            centro_x=cx_r, centro_y=cy_r)
    if(k == 'm'):
        save_axes("axes_output_m.csv", row)
    else:
        save_axes("axes_output_r.csv", row)

    return

def Convex_Hull(bin_img, tol=8, ksize_open=3):
    """
    Draw the convex hull of the largest external contour on both images and
    apply a morphological opening only on the hull border (inner tolerance = `tol` pixels).

    Parameters
    ----------
    bin_img : np.ndarray
        Binary image (modified in place).
    gray_img : np.ndarray
        Grayscale image (modified in place).
    tol : int, optional
        Internal tolerance in pixels: thickness of the 'border band' where opening is applied.
    ksize_open : int, optional
        Kernel size (square) for the morphological opening.

    Returns
    -------
    tuple
        (gray_img, bin_img) with the hull drawn and border cleaned.
    """
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # contours --> list of numpy arrays of bounding coordinates
    if not contours:
        print("No contours found.")
        return bin_img
    if len(contours) > 1:
        if cv2.contourArea(contours[0]) > cv2.contourArea(contours[1]):
            hull = cv2.convexHull(contours[0])
        else:
            hull = cv2.convexHull(contours[1])
    else:
        hull = cv2.convexHull(contours[0])
     
    cv2.drawContours(bin_img, [hull], 0, 255, thickness=8, lineType=cv2.LINE_8) 
    mask_hull = np.zeros_like(bin_img)
    cv2.drawContours(mask_hull, [hull], -1, 255, thickness=cv2.FILLED)

    kernel_er = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_inner = cv2.erode(mask_hull, kernel_er, iterations=tol)
    border_mask = cv2.subtract(mask_hull, mask_inner)  # banda di spessore ≈ tol

    # ---- 5. Opening solo su border_mask -----------------------------
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize_open, ksize_open))
    dilated = cv2.dilate(bin_img, kernel_open, iterations=5)

    #   sovrascrivo soltanto dove border_mask == 255
    bin_img[border_mask == 255] = dilated[border_mask == 255]
    return bin_img
