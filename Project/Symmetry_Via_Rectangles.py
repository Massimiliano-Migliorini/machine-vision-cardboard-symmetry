# Import main libraries: OpenCV for computer vision, NumPy for array operations, pandas for tabular data, etc.
import cv2
import numpy as np
import pandas as pd
import os
import math
from pprint import pprint

# Load configuration (tolerances, dimensions in mm, etc.) from config.yaml
from Utils import load_config_yaml, axis_from_csv, visualization, too_close


config = load_config_yaml("config.yaml")

RS_x = config["RS_x"]
RS_y = config["RS_y"]

position_mm = config["position_mm"]
length_mm = config["length_mm"]
theta_th = config["theta_degree"]

# Compute pixel thresholds (c_th_x_px, c_th_y_px, w_th_px, h_th_px) from values in mm
c_th_x_px = position_mm / RS_x 
c_th_y_px = position_mm / RS_y

w_th_px = length_mm / RS_x 
h_th_px = length_mm / RS_y 


# ------------ FUNCTIONS -----------------------------------

def pop_match(lst, cx_reference, cy_reference): 
    # Searches and removes from the list `lst` a feature with relative coordinates 
    # (cx_rel, cy_rel) close to (cx_reference, cy_reference) within a tolerance.
    # If a match is found, it is removed from `lst` and returned.
    # Otherwise, returns None.
    for i, feat in enumerate(lst): # itero su tutte le feature della lista lst
        if (abs(abs(feat["cx_rel"]) - abs(cx_reference)) <= c_th_x_px and
            abs(abs(feat["cy_rel"]) - abs(cy_reference)) <= c_th_y_px):
            return lst.pop(i)
    return None

# --- auxiliary function to validate the position on axes/center ---
def check_center(feat, start_sector):
    """
    For sectors 5,6   (Y axis):        returns True if |cy_rel| ≤ tol
    For sectors 7,8   (X axis):        returns True if |cx_rel| ≤ tol
    For sector 9      (center):        returns True if both |cx_rel| and |cy_rel| ≤ tol
    Never called for sectors 1–4.
    """
    cx, cy = feat["cx_rel"], feat["cy_rel"]
    if start_sector in (5, 6):
        return abs(cx) <= c_th_x_px

    elif start_sector in (7, 8):
        return abs(cy) <= c_th_y_px

    else:  # start_sector == 9
        return abs(cx) <= c_th_x_px and abs(cy) <= c_th_y_px

# --- auxiliary function to check if two tuples are close ---
# t1 and t2 are tuples (w, h, theta) representing two features
# w_th, h_th, theta_th are the tolerances for width, height, and angle
# Returns True if the two tuples are close within the specified tolerances


# Tuple comparison function
def is_close_tuple(t1, t2):
    """
    Compares two tuples in the form (sector, w, h, theta):
      sector: integer 1–9
      w, h in pixels; theta in degrees

    - For quadrants (1–4):
        • if features differ by only one axis (reflection on X or Y) → expected theta = -theta1
        • if they differ on both axes (reflection on X and Y) → expected theta = +theta1
        • (same sector → +theta1)
    - For sectors 5–6 or 7–8: expected theta = -theta1
    - For center (9): expected theta = +theta1
    """
    sector1, w1, h1, th1 = t1
    sector2, w2, h2, th2 = t2

    # fin th_expected
    if sector1 in (1, 2, 3, 4):
        # sector map → x and y sign
        sx1 =  1 if sector1 in (1, 4) else -1
        sy1 =  1 if sector1 in (1, 2) else -1
        sx2 =  1 if sector2 in (1, 4) else -1
        sy2 =  1 if sector2 in (1, 2) else -1

        # count how many coordinates change sign
        flips = int(sx1 != sx2) + int(sy1 != sy2)
        if flips == 1:
            th_expected = -th1
        else:
            th_expected =  th1

    elif sector1 in (5, 6, 7, 8):
        th_expected = -th1

    else:  # sector1 == 9
        th_expected =  th1

    #“fuzzy”  comparison of w, h e theta
    return (
        abs(w1 - w2)        <= w_th_px  and
        abs(h1 - h2)        <= h_th_px  and
        abs(th_expected - th2) <= theta_th
    )

# ---  “fuzzy” comparison for each single value ---
def is_close_val(e1, e2, dim):
    """
    Compares two values e1 and e2, each in the form (sector, value).
    For dim "w" or "h": absolute difference ≤ threshold.
    For dim "theta": applies flip logic based on sectors.
    """
    sector1, v1 = e1
    sector2, v2 = e2

    if dim == "w":
        return abs(v1 - v2) <= w_th_px

    elif dim == "h":
        return abs(v1 - v2) <= h_th_px

    else:  # dim == "theta"
        # 1) if within quadrants 1–4: count X/Y reflections
        if sector1 in (1,2,3,4):
            sx1 =  1 if sector1 in (1,4) else -1
            sy1 =  1 if sector1 in (1,2) else -1
            sx2 =  1 if sector2 in (1,4) else -1
            sy2 =  1 if sector2 in (1,2) else -1
            flips = int(sx1 != sx2) + int(sy1 != sy2)
            th_exp = -v1 if flips == 1 else v1

        # 2) Y axis (5–6) or X axis (7–8): always opposite
        elif sector1 in (5,6,7,8):
            th_exp = -v1

        # 3) center (9): same sign
        else:
            th_exp = v1

        return abs(th_exp - v2) <= theta_th



# --- Clustering “fuzzy” one-dimensional on the values ---
def fuzzy_groups(vals, dim):
    """
    Fuzzily groups a list of elements `vals`, each in the form (sector, value), where:
      - sector: integer from 1–9 indicating the sector the value comes from
      - value: the numerical value to be grouped (w, h, or theta)

    Args:
        vals:        list of tuples (sector, value) for the specified dimension
        dim:         string in {"w", "h", "theta"} indicating 
                     which quantity is being grouped

    Returns:
        reps:        list of cluster representatives, 
                     each always a tuple (sector, value)
        assignments: list of integers, where assignments[i] 
                     is the cluster index of vals[i]
        counts:      list of counts for each cluster
    """


    reps = []         # List of cluster representatives
    assignments = []  # assignments[i] = index of the cluster to which vals[i] belongs

    # Iterate on each element vals[i]
    for e in vals:
        placed = False

        # Try to insert it into an existing cluster
        for gi, rep in enumerate(reps):
            if is_close_val(rep, e, dim):
                # If "close" to the representative `rep`, assign `e` to that cluster
                assignments.append(gi)
                placed = True
                break

        # If not close to any representative, create a new cluster
        if not placed:
            reps.append(e)
            assignments.append(len(reps) - 1)

    # Count how many elements are in each cluster
    counts = [assignments.count(i) for i in range(len(reps))]

    return reps, assignments, counts


def check_group(feats):
    """
    Returns "OK" if all features in `feats` have status_feature == "symmetric",
    otherwise returns "KO".
    """
    if all(f["status_feature"] == "symmetric" for f in feats):
        return "OK"
    else:
        return "KO"
        

# axial_symmetry_detection_r: entry point for detecting axial symmetry based on minimum rectangles
def axial_symmetry_detection_r(bin_img, img_path, gray_img):

    # Convert grayscale image to BGR to allow drawing colored annotations
    img_color = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    config = load_config_yaml("config.yaml")

    w, h = gray_img.shape[:2]
    ratio = h / w 

    AREA_SLOT_MIN_R = config['AREA_SLOT_MIN_R']
    AREA_SLOT_MAX_R = config['AREA_SLOT_MAX_R']
    ASPECT_SLOT_MIN = config['ASPECT_SLOT_MIN']
    ASPECT_SLOT_MAX = config['ASPECT_SLOT_MAX']
    TOL_RADIUS = config["TOL_RADIUS"]
    INTENSITY_HOLE_MAX_R = config["INTENSITY_HOLE_MAX_R"]

    # Read symmetry center coordinates from the previously generated CSV file
    cx_r, cy_r = axis_from_csv(img_path, "axes_output_r.csv")

    accepted_centers = []
    features = []

    # Extract all contours from the binary image
    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # For each contour: compute minimum bounding rectangle, area, aspect ratio, and apply intensity filtering
    for c in contours:

        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), theta = rect

        if theta >= 45.0:
            theta -= 90
            rw, rh = rh, rw

        area   = rw * rh
        aspect = (rw/rh) if rh < rw else (rh/rw)

        box  = cv2.boxPoints(rect).astype(np.int32)
        mask = np.zeros_like(gray_img, dtype=np.uint8)
        cv2.fillPoly(mask, [box], 255)
        mean_int = cv2.mean(gray_img, mask=mask)[0]
        
        if mean_int > INTENSITY_HOLE_MAX_R:
            continue
        
        if too_close((cx, cy), accepted_centers, TOL_RADIUS**2):
            continue

        accepted_centers.append((cx, cy))

        
        if not (ASPECT_SLOT_MIN < aspect < ASPECT_SLOT_MAX and
                AREA_SLOT_MIN_R  < area   < AREA_SLOT_MAX_R):
            continue

        features.append({
            "cx":    cx,
            "cy":    cy,
            "w":     rw,
            "h":     rh,
            "theta": theta,
            
            "w_valid":     True,
            "h_valid":     True,
            "theta_valid": True,
            "cx_valid":    True,
            "cy_valid":    True,
            "status_feature": None
        })
   
    defected  = False      # Becomes True if we find any outlier / null / missing feature

    feature_sector_1, feature_sector_2 = [], [] # alto a dx, alto a sx
    feature_sector_3, feature_sector_4 = [], [] # basso a sx, basso a dx
    feature_on_x_left = [] 
    feature_on_x_right = []
    feature_on_y_down = []
    feature_on_y_up = [] 
    feature_on_xy = [] 

    # Compute coordinates relative to the center (cx_rel, cy_rel) and assign to corresponding geometric sectors
    for f in features:
      
        cx_rel = f["cx"] - cx_r
        cy_rel = cy_r - f["cy"]
        f["cx_rel"], f["cy_rel"] = cx_rel, cy_rel 

        w2 = f["w"] / 2 
        h2 = f["h"] / 2
        theta = -math.radians(f["theta"])  
       
        delta_x = abs(w2 * math.cos(theta)) + abs(h2 * math.sin(theta)) 
        delta_y = abs(w2 * math.sin(theta)) + abs(h2 * math.cos(theta)) 

        if abs(cx_rel) <= delta_x and abs(cy_rel) <= delta_y:
            feature_on_xy.append(f)

        elif abs(cx_rel) <= delta_x and cy_rel < 0:
            feature_on_y_down.append(f)
        # Y axis pointing upwards
        elif abs(cx_rel) <= delta_x and cy_rel > 0:
            feature_on_y_up.append(f)
        # X axis pointing to the left
        elif abs(cy_rel) <= delta_y and cx_rel < 0:
            feature_on_x_left.append(f)
        # X axis pointing to the right
        elif abs(cy_rel) <= delta_y and cx_rel > 0:
            feature_on_x_right.append(f)
        
        elif cx_rel >= 0 and cy_rel >= 0:
            feature_sector_1.append(f)
        elif cx_rel <  0 and cy_rel >= 0:
            feature_sector_2.append(f)
        elif cx_rel <  0 and cy_rel <  0:
            feature_sector_3.append(f)
        # Y axis pointing downwards
        else:
            feature_sector_4.append(f)

    sector_lists = {
        1: feature_sector_1,
        2: feature_sector_2,
        3: feature_sector_3,
        4: feature_sector_4,
        5: feature_on_y_up,
        6: feature_on_y_down,
        7: feature_on_x_right,
        8: feature_on_x_left,
        9: feature_on_xy
    }

    slots_dict = {}
    slot_index = 0

    
    # Iterative grouping: for each reference feature, search for matching features in other sectors
    for start_sector in range(1, 10):
        while sector_lists[start_sector]:
            ref = sector_lists[start_sector].pop(0)
            x_ref, y_ref = ref["cx_rel"], ref["cy_rel"]
            

            found = {start_sector: ref} 
            
            if start_sector in (1,2,3,4):
                for s in (1,2,3,4):
                    if s==start_sector: continue 
                    m = pop_match(sector_lists[s], x_ref, y_ref)
                    if m: found[s] = m
            elif start_sector in (5,6):
                for s in (5,6):
                    if s==start_sector: continue
                    m = pop_match(sector_lists[s], x_ref, y_ref)
                    if m: found[s] = m
            elif start_sector in (7,8):
                for s in (7,8):
                    if s==start_sector: continue
                    m = pop_match(sector_lists[s], x_ref, y_ref)
                    if m: found[s] = m
            else:
                pass 
            
            feats = [found[s] for s in sorted(found)]
            
            keys = ("w", "h", "theta")

            tuples = [(s, feat["w"], feat["h"], feat["theta"]) for s, feat in zip(sorted(found), feats)]
  
            groups = []
            assignments = []

            # Iterative grouping: for each reference feature, search for matches in the other sectors
            for t in tuples:
                placed = False
                for gi, rep in enumerate(groups):
                    if is_close_tuple(t, rep):       
                        assignments.append(gi)
                        placed = True
                        break
                if not placed:
                    groups.append(t)
                    assignments.append(len(groups) - 1)


            u = len(groups) 
            counts = [assignments.count(i) for i in range(u)]
            n_feats = len(feats)


            if start_sector in (5, 6, 7, 8):
                
                if len(feats) == 2 and is_close_tuple(tuples[0], tuples[1]):
                    ok0 = check_center(feats[0], start_sector)
                    ok1 = check_center(feats[1], start_sector)
                    if ok0 and ok1:
                        statuses = ["symmetric", "symmetric"]
                    else:
                        statuses = ["half", "half"]
                else:
                    statuses = ["outlier"] * len(feats)

            elif start_sector == 9:
                feat = feats[0]
                statuses = ["symmetric"] if check_center(feat, start_sector) else ["outlier"]
            else:
                if u == 1:
                    if   n_feats == 4: statuses = ["symmetric"] * 4
                    elif n_feats == 3: statuses = ["missing"]   * 3
                    elif n_feats == 2: statuses = ["half"]      * 2
                    else:               statuses = ["outlier"]
                else:
                    if n_feats == 4:
                        if u == 2:
                            if 3 in counts:
                                major = counts.index(3)
                                statuses = [
                                    "missing" if assignments[i]==major else "outlier"
                                    for i in range(4)
                                ]
                            else:
                                statuses = ["half"] * 4

                        elif u == 3:
                            statuses = [
                                "half"   if counts[assignments[i]]==2 else "outlier"
                                for i in range(4)
                            ]
                        else: 
                            statuses = ["outlier"] * 4
                    elif n_feats == 3:
                        if u == 2:
                            statuses = [
                                "half"   if counts[assignments[i]]==2 else "outlier"
                                for i in range(3)
                            ]
                        else:
                            statuses = ["outlier"] * 3
                    elif n_feats == 2:
                        if u == 2:
                            statuses = ["outlier"] * 2
                        else:
                            statuses = ["half"] * 2
                    else:  
                        statuses = ["outlier"]
            
            if len(feats) == 1 and statuses[0] == "outlier":
                feats[0]["cx_valid"] = False
                feats[0]["cy_valid"] = False

            for feat, st in zip(feats, statuses):
                feat["status_feature"] = st

            for dim in ("w", "h", "theta"):
                sectors = sorted(found)  
                vals = [(s, feat[dim]) for s, feat in zip(sectors, feats)] 
                reps, assignments, counts = fuzzy_groups(vals, dim)

                if len(reps) == 1:
                    for feat in feats:
                        feat[f"{dim}_valid"] = True

                else:
                    max_count = max(counts)

                    if counts.count(max_count) > 1:
                        for feat in feats:
                            feat[f"{dim}_valid"] = False

                    else:
                        maj_group = counts.index(max_count)

                        for feat, grp in zip(feats, assignments):
                            feat[f"{dim}_valid"] = (grp == maj_group)

            
            status = check_group(feats)
            defected |= (status != "OK") 

            slot_index += 1
            slot_name = f"slot_{slot_index}"

            slots_dict[slot_name] = [status] + feats

    
    print("\n=== RESULTS OF SYMMETRY CHECK USING RECTANGLES===")
    print(f"Created {slot_index} slot:")
    for name, content in slots_dict.items():
        print(f"  {name}: {content[0]} ({len(content)-1} feature)")

    print(f"Defected Cardboard? {'Yes' if defected else 'No'}")

    
    print("\n--- Features list  ---")
    for idx, feat in enumerate(features, start=1):
        print(f"feature {idx}:")
        pprint(feat, width=80)
        print() 



    # ---------- RESULT VISUALIZATION -----------------------------------
    # VISUALIZATION: draw boxes, measurement annotations, and feature status
    w_disp = 800
    h_disp = int(800 / ratio)  

    base_ref     = 1000.0           
    scale        = w / base_ref     
    font         = cv2.FONT_HERSHEY_SIMPLEX
    font_scale   = 0.7* scale
    thick_border = max(1, int(round(3 * scale)))
    thick_text   = max(1, int(round(1 * scale)))
    line_type    = cv2.LINE_AA
    y_offset     = int(round(60 * scale))  
    line_spacing = int(round(25 * scale))  
    color_map = {
        "symmetric": (  0, 255,   0),
        "missing"  : (  0, 100,   0),
        "half"     : (  0, 128, 255),
        "outlier"  : (  0,   0, 255),
    }

    for f in features:
        # Drawing the rectangle and text labels (dimensions in mm, angle, relative position)
        cx, cy, w_, h_, th = f["cx"], f["cy"], f["w"], f["h"], f["theta"]
        cx_rel_mm = f["cx_rel"] * RS_x
        cy_rel_mm = f["cy_rel"] * RS_y

        status     = f.get("status_feature", "outlier")
        rect_color = color_map.get(status, (0, 0, 255))

        box = cv2.boxPoints(((cx, cy), (w_, h_), th))
        box = np.intp(box)
        cv2.polylines(img_color, [box], True, rect_color, 2)

        # w_mm = w_ * RS_x
        # h_mm = h_ * RS_y

        # line1 = [
        #     (f"{w_mm:.2f}mm", (255,255,255) if f["w_valid"]     else (0,0,255)),
        #     (" x ",           (255,255,255)),
        #     (f"{h_mm:.2f}mm", (255,255,255) if f["h_valid"]     else (0,0,255)),
        # ]
        # line2 = [
        #     (f"th={th:.2f} deg",   (255,255,255) if f["theta_valid"] else (0,0,255)),
        # ]

        # line_color = (255,255,255) if (f["cx_valid"] and f["cy_valid"]) else (0,0,255)
        # line3 = [
        #     (f"({cx_rel_mm:.1f} mm,{cy_rel_mm:.1f} mm)", line_color),
        # ]

        # total_w1 = sum(cv2.getTextSize(seg, font, font_scale, thick_text)[0][0] for seg,_ in line1)
        # x_txt1   = int(cx - total_w1 / 2)
        # total_w2 = sum(cv2.getTextSize(seg, font, font_scale, thick_text)[0][0] for seg,_ in line2)
        # x_txt2   = int(cx - total_w2 / 2)
        # total_w3 = sum(cv2.getTextSize(seg, font, font_scale, thick_text)[0][0] for seg,_ in line3)
        # x_txt3   = int(cx - total_w3 / 2)

        # y_top = int(box[np.argmin(box[:,1]), 1])
        # y_txt = y_top - y_offset

        # def draw_segmented_line(segments, x0, y0):
        #     x = x0
        #     for txt, clr in segments:
        #         cv2.putText(img_color, txt, (x, y0),
        #                     font, font_scale,
        #                     (0, 0, 0), thick_border, line_type)
        #         cv2.putText(img_color, txt, (x, y0),
        #                     font, font_scale,
        #                     clr, thick_text, line_type)
        #         x += cv2.getTextSize(txt, font, font_scale, thick_text)[0][0]

        # draw_segmented_line(line1, x_txt1,       y_txt)
        # draw_segmented_line(line2, x_txt2, y_txt +   line_spacing)
        # draw_segmented_line(line3, x_txt3, y_txt + 2*line_spacing)

    # Save debug image and return the annotated image
    cv2.imwrite("immagini/debug/IMG_17.png", img_color)
    return img_color
