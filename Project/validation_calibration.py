import cv2 as cv
import os

CHESS_BOARD_DIM = (8, 5)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ---------------------- VERIFICA IMMAGINI CALIBRAZIONE ----------------------

def verify_calibration_images(src_folder, dst_folder, board_dim, criteria):
    os.makedirs(dst_folder, exist_ok=True)
    valid_count = 0

    for fname in os.listdir(src_folder):
        fpath = os.path.join(src_folder, fname)

        img = cv.imread(fpath)
        if img is None:
            print(f"[ERRORE] Immagine non caricata: {fname}")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, board_dim, None)
        if ret:
            corners_refined = cv.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
            img_marked = cv.drawChessboardCorners(img.copy(), board_dim, corners_refined, ret)

            out_path = os.path.join(dst_folder, fname)
            cv.imwrite(out_path, img_marked)
            valid_count += 1
            print(f"[OK] Scacchiera rilevata: {fname}")
        else:
            print(f"[NO] Nessuna scacchiera trovata: {fname}")

    print(f"\nTotale immagini valide salvate in '{dst_folder}': {valid_count}")

# Esegui la verifica delle immagini salvate precedentemente
verify_calibration_images(
    src_folder="c_images",
    dst_folder="calib_validated",
    board_dim=CHESS_BOARD_DIM,
    criteria=criteria
)
