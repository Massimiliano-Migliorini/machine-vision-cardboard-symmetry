import cv2
import numpy as np
import csv, os, math
import yaml
import pandas as pd




def load_config_yaml(filepath):
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

def too_close(new_pt, accepted_pts, tol2): 
    config = load_config_yaml("config.yaml")

    for (x, y) in accepted_pts:
        if (new_pt[0]-x)**2 + (new_pt[1]-y)**2 < tol2:
            return True
    return False

def save_axes(csv_path: str, row: dict) -> None:  
    """
    Append or update a row in the axes CSV file.

    If an entry with the same image name exists, it is updated; otherwise, a new row is added.

    Parameters:
        csv_path (str): Path to the CSV file.
        row (dict): Dictionary containing keys like 'image', 'centro_x', and 'centro_y'.
    """
    rows, updated = [], False

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["image"] == row["image"]:
                    rows.append(row)
                    updated = True
                else:
                    rows.append(r)

    if not updated:
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'aggiornata' if updated else 'salvata'} '{row['image']}'")


def axis_from_csv(img_path, csv_path):

    df_axes = pd.read_csv(csv_path)
    img_name = os.path.basename(img_path)          

    row = df_axes.loc[df_axes["image"] == img_name]
    if row.empty:
        raise ValueError(f"{img_name} non trovato in {csv_path}")

    cx_ax = float(row.iloc[0]["centro_x"])
    cy_ax = float(row.iloc[0]["centro_y"])
    return (cx_ax, cy_ax)

def visualization(img):
    """
    Display an image in a resizable OpenCV window.

    Parameters:
        img (np.ndarray): Image to display (grayscale or BGR).
    """
    w, h = img.shape[:2]
    ratio = h / w 
    w_disp = 800
    h_disp = int(800 * ratio)
    
       

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", w_disp, h_disp)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return
