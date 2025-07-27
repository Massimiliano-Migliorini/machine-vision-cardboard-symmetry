import cv2
import numpy as np
import csv, os, math
import pandas as pd
import yaml

from Utils import load_config_yaml, axis_from_csv, too_close

config = load_config_yaml("config.yaml")

RS_x = config["RS_x"]
RS_y = config["RS_y"]

position_mm = config["position_mm"]

AREA_MIN_M = config["AREA_MIN_M"]
AREA_MAX_M = config["AREA_MAX_M"]

TOL_RADIUS = config["TOL_RADIUS"]

INTENSITY_HOLE_MAX_M = config["INTENSITY_HOLE_MAX_M"]

c_th_x_px = position_mm / RS_x 
c_th_y_px = position_mm / RS_y



def _pop_match(lst, cx_reference, cy_reference):
    """
    Searches for and removes the first feature in a list that matches a reference position
    within a specified tolerance.

    Parameters:
        lst (list of dict): A list of feature dictionaries, each containing 'cx_rel' and 'cy_rel'.
        cx_reference (float): Reference x-coordinate (relative) to match against.
        cy_reference (float): Reference y-coordinate (relative) to match against.

    Returns:
        dict or None: The matched and removed feature dictionary, or None if no match is found.
    """
    for i, feat in enumerate(lst): 
        if (abs(abs(feat["cx_rel"]) - abs(cx_reference)) <= c_th_x_px and
            abs(abs(feat["cy_rel"]) - abs(cy_reference)) <= c_th_y_px):
            return lst.pop(i)
    return None




def check_symmetry(contour, thr=0.05):
    """
    Checks the horizontal and vertical symmetry of an Nx1x2 contour
    and applies a threshold to decide if it is 'symmetric'.

    Args:
        contour (np.ndarray): Contour of shape NumPoints x 1 x 2 of (x, y) points.
        thr (float): Maximum acceptable asymmetry threshold (between 0 and 1).

    Returns:
        dict: {
            'asymmetry_lr': float,  # horizontal asymmetry index
            'asymmetry_ud': float,  # vertical asymmetry index
            'is_symmetric': bool    # True if both indices are ≤ thr
        }
    """
    cnt = np.asarray(contour, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(cnt)
    cnt_shifted = cnt - np.array([[[x, y]]])

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [cnt_shifted], 255)

    mask_lr = np.fliplr(mask)
    mask_ud = np.flipud(mask)

    diff_lr = cv2.bitwise_xor(mask, mask_lr)
    diff_ud = cv2.bitwise_xor(mask, mask_ud)

    filled_pixels = np.count_nonzero(mask)
    asym_lr = np.count_nonzero(diff_lr) / float(filled_pixels)
    asym_ud = np.count_nonzero(diff_ud) / float(filled_pixels)

    return (asym_lr <= thr) and (asym_ud <= thr)


def _check_position(feat, start_sector): 
    """
    For sectors 5,6   (Y axis):        returns True if |cy_rel| ≤ tol
    For sectors 7,8   (X axis):        returns True if |cx_rel| ≤ tol
    For sector 9      (center):        returns True if both |cx_rel| and |cy_rel| ≤ tol
    Never called for sectors 1–4.
    """

    if start_sector in (5, 6):
        return abs(feat["cx_rel"]) <= c_th_x_px

    elif start_sector in (7, 8):
        return abs(feat["cy_rel"]) <= c_th_y_px

    else:  
        return abs(feat["cx_rel"]) <= c_th_x_px and abs(feat["cy_rel"]) <= c_th_y_px and check_symmetry(feat['cnt'])


def _is_close_tuple(t1, t2, moments_params_path="config.yaml"): 
    """
    Compare two moment tuples for approximate equality.

    Each tuple has the form:
        ( sector,
          m00, m10, m01,
          m20, m11, m02,
          m30, m21, m12, m03 )

    This function checks whether all corresponding raw moments
    differ by at most a configured relative threshold. The threshold
    'm_th_rel' is loaded from the YAML file specified by
    'moments_params_path'. Returns True if every moment pair
    satisfies:
        abs(v1 - v2) / min(v1, v2) <= m_th_rel
    Otherwise returns False.
    """
    moments1 = t1[1:]
    moments2 = t2[1:]

    config = load_config_yaml(moments_params_path)
    th_rel = config["m_th_rel"]
    
    for v1, v2 in zip(moments1, moments2):
        print(f"sector:{t1[0]}, scarto relativo {100*abs(v1 - v2)/min(v1,v2)}%")
        if min(v1, v2) == 0 or abs(v1 - v2)/min(v1,v2) > th_rel:
            return False
    return True

def _check_group(feats):
    """
    Returns "OK" if all features in `feats` have status_feature == "symmetric",
    otherwise returns "KO".
    """
    if all(f["status_feature"] == "symmetric" for f in feats):
        return "OK"
    else:
        return "KO"


def _raw_moments_up_to_third_order(contour, image_shape):
    """
    Computes all raw spatial moments up to the third order for a given contour.

    The method generates a binary mask of the same shape as the input image, 
    where the specified contour is drawn and filled in white (255) on a black 
    background (0). This mask serves to isolate the region enclosed by the 
    contour, allowing accurate calculation of geometric moments via OpenCV's 
    `cv2.moments` function.

    Parameters:
        contour (np.ndarray): An array of (x, y) coordinates defining the contour.
        image_shape (tuple): The shape of the image (height, width) used to size the mask.

    Returns:
        dict: A dictionary containing raw spatial moments M_pq for all p + q ≤ 3.
              Specifically, the keys returned are:
              ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03'].
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], contourIdx=-1, color=255, thickness=cv2.FILLED)
    # Show the binary mask for each contour
    # cv2.imshow("Contour Mask", mask)
    # cv2.waitKey(0)

    M = cv2.moments(mask)

    return {
        'm00': M['m00'],      
        'm10': M['m10'],      
        'm01': M['m01'],      
        'm20': M['m20'],      
        'm11': M['m11'],      
        'm02': M['m02'],    
        'm30': M['m30'],     
        'm21': M['m21'],     
        'm12': M['m12'],      
        'm03': M['m03'],      
    }

def _anti_reflection(sector, cnt, cx_ax, cy_ax):
    """
    Reflect a given contour into Sector 1 (top-right quadrant) based on its original sector.

    This function mirrors a contour across the appropriate axis so that all features,
    regardless of their original quadrant (sector 1–4), are aligned into Sector 1.
    The reflection is performed relative to the symmetry center (cx_ax, cy_ax), which
    serves as the origin of the quadrant system.

    Parameters:
        sector (int): Original sector (1 to 4) of the contour.
        cnt (np.ndarray): Contour as returned by cv2.findContours, with shape (N, 1, 2).
        cx_ax (float): X-coordinate of the symmetry center (vertical axis).
        cy_ax (float): Y-coordinate of the symmetry center (horizontal axis).

    Returns:
        np.ndarray: The contour reflected into Sector 1, with the same shape (N, 1, 2).
    """
    cnt_new = cnt.copy()

    for i in range(cnt.shape[0]):
        x, y = cnt[i, 0]

        # Convert to coordinates relative to symmetry center
        x_rel = x - cx_ax
        y_rel = cy_ax - y  # y-axis is top-down in image coordinates

        # Apply reflection based on sector
        if sector == 1 or sector == 9 or sector == 6 or sector == 8:
            x_new_rel = x_rel
            y_new_rel = y_rel
        elif sector == 2:
            x_new_rel = -x_rel
            y_new_rel = y_rel
        elif sector == 3:
            x_new_rel = -x_rel
            y_new_rel = -y_rel
        elif sector == 4:
            x_new_rel = x_rel
            y_new_rel = -y_rel
        elif sector == 5: 
            x_new_rel = x_rel
            y_new_rel = -y_rel
        elif sector == 7:
            x_new_rel = -x_rel
            y_new_rel = -y_rel

        else:
            raise ValueError(f"Invalid sector: {sector}")

        # Convert back to absolute image coordinates
        x_new = x_new_rel + cx_ax
        y_new = cy_ax - y_new_rel

        cnt_new[i, 0] = [x_new, y_new]

    return cnt_new




def axial_symmetry_detection_m(bin_img, img_path, gray_img):
    """
    Description:

    Each dictionary has:
        - "moments": moments up to the 3rd order of the white pixels in the contour.
        - "centroid": (cx, cy) if area > 0, else None
        - "group": 
        - "sector": 

    Parameters:
        bin_img (np.ndarray): Binary input image.

    Returns:
    """
    img_color = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    img_color_2= img_color.copy()  # Copy to draw contours later

    cx_ax, cy_ax = axis_from_csv(img_path, "axes_output_m.csv")
    
    accepted_centers = []      
    features = []

    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        mask = np.zeros_like(bin_img, dtype=np.uint8)
        cv2.fillPoly(mask, [cnt], 255)
        mean_int = cv2.mean(bin_img, mask=mask)[0]
        # if the mean intensity is too high (light gray of the cardboard), discard
        if mean_int > INTENSITY_HOLE_MAX_M:
            continue
        moments = _raw_moments_up_to_third_order(cnt, bin_img.shape)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]

            if too_close((cx, cy), accepted_centers, TOL_RADIUS**2):
                continue
            accepted_centers.append((cx, cy))
        else:
            continue

        cx_rel = cx - cx_ax
        cy_rel = cy_ax - cy

        pts = cnt[:, 0, :]  

        x_min = np.min(pts[:, 0])
        x_max = np.max(pts[:, 0])
        y_min = np.min(pts[:, 1])
        y_max = np.max(pts[:, 1])

        x_sx_rel = x_min - cx_ax
        x_dx_rel = x_max - cx_ax

        y_down_rel = cy_ax - y_max  
        y_up_rel = cy_ax - y_min  


        if (AREA_MIN_M < moments['m00'] < AREA_MAX_M):
            cv2.drawContours(img_color, [cnt], 0, (0,255,0), 2)
            if x_sx_rel <= 0 <= x_dx_rel and y_down_rel <= 0 <= y_up_rel:
                features.append({"S": 9})
            elif x_sx_rel <= 0 <= x_dx_rel and y_up_rel < 0:
                features.append({"S": 6}) #6
            elif x_sx_rel <= 0 <= x_dx_rel and y_down_rel > 0:
                features.append({"S": 5}) #5
            elif y_down_rel <= 0 <= y_up_rel and x_dx_rel < 0:
                features.append({"S": 8}) #8
            elif y_down_rel <= 0 <= y_up_rel and x_sx_rel > 0:
                features.append({"S": 7}) #7
            elif cx_rel >= 0 and cy_rel >= 0:
                features.append({"S": 1})
            elif cx_rel < 0 and cy_rel > 0:
                features.append({"S": 2})
            elif cx_rel < 0 and cy_rel < 0:
                features.append({"S": 3})
            else:
                features.append({"S": 4})
            features[-1].update({"cnt": cnt})

            cnt = _anti_reflection(features[-1]["S"], cnt, cx_ax, cy_ax)  
            moments = _raw_moments_up_to_third_order(cnt, bin_img.shape)

            features[-1].update({
                # Centroid in axes of symmetry coordinates
                "cx_rel":    cx_rel, # in [pixel]
                "cy_rel":    cy_rel, # in [pixel]
                # Moments up to the 3rd order, expressed assuming the feature in sector 1, 5 or 7, to rapresent all the sampled features in a Template reference system
                "M": moments, 
                # Symmetry status of the feature
                "status_feature": None
            })
            cv2.drawContours(img_color_2, [cnt], 0, (0,255,0), 2)

    defected  = False     

    feature_sector_1, feature_sector_2 = [], [] 
    feature_sector_3, feature_sector_4 = [], []
    feature_on_x_left = [] 
    feature_on_x_right = []
    feature_on_y_down = []
    feature_on_y_up = [] 
    feature_on_xy = [] 

    for f in features:
        if f["S"] == 1:
            feature_sector_1.append(f)
        elif f["S"] == 2:
            feature_sector_2.append(f)
        elif f["S"] == 3:
            feature_sector_3.append(f)
        elif f["S"] == 4:
            feature_sector_4.append(f)
        elif f["S"] == 5:
            feature_on_y_up.append(f)
        elif f["S"] == 6:
            feature_on_y_down.append(f)
        elif f["S"] == 7:
            feature_on_x_right.append(f)
        elif f["S"] == 8:
            feature_on_x_left.append(f)
        elif f["S"] == 9:
            feature_on_xy.append(f)
        
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

    slot_dict = {}
    slot_index = 0

    for start_sector in range(1, 10):
        while sector_lists[start_sector]:
            ref = sector_lists[start_sector].pop(0)

            found = {start_sector: ref} 

            if start_sector in (1,2,3,4):
                for s in (1,2,3,4):
                    if s==start_sector: continue
                    m = _pop_match(sector_lists[s], ref["cx_rel"], ref["cy_rel"])
                    if m: found[s] = m
            elif start_sector in (5,6):
                for s in (5,6):
                    if s==start_sector: continue
                    m = _pop_match(sector_lists[s],ref["cx_rel"], ref["cy_rel"])
                    if m: found[s] = m
            elif start_sector in (7,8):
                for s in (7,8):
                    if s==start_sector: continue
                    m = _pop_match(sector_lists[s], ref["cx_rel"], ref["cy_rel"])
                    if m: found[s] = m
            else:
                pass 
            
            
            feats = [found[s] for s in sorted(found)] 
            
            tuples = []
            for s, feat in zip(sorted(found), feats):
                M = feat["M"]
                tuples.append((
                    s,
                    M['m00'],  M['m10'],  M['m01'],
                    M['m20'],  M['m11'],  M['m02'],
                    M['m30'],  M['m21'],  M['m12'],  M['m03']
                ))
            
            groups = []
            assignments = []
            for t in tuples:
                placed = False
                for gi, rep in enumerate(groups):
                    if _is_close_tuple(t, rep):
                        assignments.append(gi)
                        placed = True
                        break
                if not placed:
                    groups.append(t)
                    assignments.append(len(groups) - 1) # each element is a placeholder indicating group membership (e.g. [0,0,0,1]: 3 elements in the first group)

            n_groups = len(groups)
            counts = [assignments.count(i) for i in range(n_groups)]
            n_feats = len(feats)

            # ————————————————————————————————————————
            # 4) Classification by sector (using is_close_tuple on moment tuples)
            # ————————————————————————————————————————

            if start_sector in (5, 6, 7, 8):
                # on the axes only, I expect at most 2 symmetric features
                if len(feats) == 2 and _is_close_tuple(tuples[0], tuples[1]):
                    ok0 = _check_position(feats[0], start_sector)
                    ok1 = _check_position(feats[1], start_sector)
                    # both truly on the axis → symmetric, otherwise half
                    statuses = (["symmetric", "symmetric"]
                                if ok0 and ok1
                                else ["half", "half"])
                else:
                    # only one or two unmatched → outlier
                    statuses = ["outlier"] * len(feats)

            elif start_sector == 9:
                # center: single expected feature, check proximity to (0,0)
                feat = feats[0]
                statuses = (["symmetric"]
                            if _check_position(feat, start_sector)
                            else ["outlier"])

            else:
                # quadrants 1–4
                # if the whole group is identical (n_groups == 1) → assign status based on number of features
                if n_groups == 1:
                    if   n_feats == 4: statuses = ["symmetric"] * 4
                    elif n_feats == 3: statuses = ["missing"]   * 3
                    elif n_feats == 2: statuses = ["half"]      * 2
                    else:               statuses = ["outlier"]

                # otherwise handle mixed cases as before
                else:
                    if n_feats == 4:
                        if n_groups == 2: 
                            # 3 vs 1
                            if 3 in counts:
                                major = counts.index(3)
                                statuses = [
                                    "missing" if assignments[i]==major else "outlier"
                                    for i in range(4)
                                ]
                            else:
                                # 2+2
                                statuses = ["half"] * 4

                        elif n_groups == 3:
                            # 1 couple + 2 singles
                            statuses = [
                                "half" if counts[assignments[i]]==2 else "outlier"
                                for i in range(4)
                            ]
                        else:  # n_groups == 4
                            statuses = ["outlier"] * 4

                    elif n_feats == 3:
                        if n_groups == 2:
                            # 2+1
                            statuses = [
                                "half" if counts[assignments[i]]==2 else "outlier"
                                for i in range(3)
                            ]
                        else:
                            statuses = ["outlier"] * 3

                    elif n_feats == 2:
                        if n_groups == 2:
                            # 1+1
                            statuses = ["outlier"] * 2
                        else:
                            # u==1 already handled above
                            statuses = ["half"] * 2

                    else:  # n_feats == 1
                        statuses = ["outlier"]

            
        


            # Assign symmetry status to each feature
            for feat, st in zip(feats, statuses):
                feat["status_feature"] = st

    

            
            # --- 3e. General classification of the feature group --> implemented as detection of values different from 'symmetric' 
            # within the group --> the classification will only result in "OK" or "KO" (defected) for the feature group. 
            # Afterwards, we update the flag for the entire cardboard, which is considered defective if at least one group is KO.
            status = _check_group(feats)
            defected |= (status != "OK") # works like a logical OR: if at least one of the feature groups has status other than "OK", then `defected` becomes True

            slot_index += 1
            slot_name = f"slot_{slot_index}"

            slot_dict[slot_name] = [status] + feats

    # ---------- RESULT VISUALIZATION -----------------------------------

    # 1) Compute display window dimensions
    h_img, w_img = img_color.shape[:2]
    ratio       = h_img / w_img
    w_disp      = 800
    h_disp      = int(w_disp / ratio)

    # 2) Compute text scaling factor
    base_ref     = 1000.0           # reference width for which font_scale = 0.45
    scale        = w_img / base_ref
    font         = cv2.FONT_HERSHEY_SIMPLEX
    font_scale   = 0.5 * scale
    thick_text   = max(1, int(round(1 * scale)))
    line_type    = cv2.LINE_AA

    # 3) BGR color mapping to draw contours based on feature status_feature
    color_map = {
        "symmetric": (  0, 255,   0),
        "missing"  : (  0, 100,   0),
        "half"     : (  0, 128, 255),
        "outlier"  : (  0,   0, 255),
    }

    # 4) Loop over all detected features
    for f in features:
        # corner case: if the original contour is not found, use feature["cnt"] which was previously saved
        cnt = f.get("cnt")
        if cnt is not None:
            status = f.get("status_feature", "outlier")
            color  = color_map.get(status, (0, 0, 255))
            # draw the colored contour
            cv2.drawContours(img_color, [cnt], -1, color, 2)

    
    print("\n=== RESULTS OF SYMMETRY CHECK USING MOMENTS===")
    for slot_name, data in slot_dict.items():
        group_status = data[0]
        features = data[1:]
        
        print(f"\n{slot_name.upper()} → GROUP STATUS: {group_status}")
        print(f"  Number of found features: {len(features)}")
        
        
        for i, feat in enumerate(features, 1):
            moments = feat['M']
            print(f"Feature {i}, Moments {moments}")
        
        for i, feat in enumerate(features, 1):
            cx = round(feat['cx_rel'], 2)
            cy = round(feat['cy_rel'], 2)
            status = feat['status_feature']
            print(f"Feature {i}: centroid=({cx}, {cy}), status={status}. sector ={feat['S']}")

    return img_color, img_color_2

