import cv2
import numpy as np

def nothing(x):
    pass

img_path = r"immagini\setup_definitivo\IMG_LINEE2.bmp"

# 1. Carica l'immagine da disco
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)

cv2.namedWindow('Canny - Parametri', cv2.WINDOW_NORMAL)

# Imposta dimensioni più ampie della finestra (es. 1200x800)
cv2.resizeWindow('Canny - Parametri', 1200, 800)

# Trackbar
cv2.createTrackbar('Blur kernel (odd)', 'Canny - Parametri', 5, 30, nothing)
cv2.createTrackbar('Sigma x0.1', 'Canny - Parametri', 10, 100, nothing)  # Moltiplicheremo per 0.1
cv2.createTrackbar('Threshold1', 'Canny - Parametri', 100, 500, nothing)
cv2.createTrackbar('Threshold2', 'Canny - Parametri', 200, 500, nothing)

while True:
    # Leggi i parametri dalle trackbar
    k = cv2.getTrackbarPos('Blur kernel (odd)', 'Canny - Parametri')
    sigma = cv2.getTrackbarPos('Sigma x0.1', 'Canny - Parametri') * 0.1
    t1 = cv2.getTrackbarPos('Threshold1', 'Canny - Parametri')
    t2 = cv2.getTrackbarPos('Threshold2', 'Canny - Parametri')

    # Assicura che il kernel sia dispari e ≥ 3
    k = max(3, k)
    if k % 2 == 0:
        k += 1

    # Applica blur gaussiano
    blurred = cv2.GaussianBlur(img_gray, (k, k), sigmaX=sigma, sigmaY=sigma)

    # Applica Canny
    edges = cv2.Canny(blurred, t1, t2)

    # Trova solo contorni esterni
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Disegna contorni su copia dell'immagine a colori
    display = img_color.copy()
    cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

    # Mostra il risultato
    cv2.imshow('Canny - Parametri', display)

    key = cv2.waitKey(1) & 0xFF

    # Salva parametri su pressione di 's'
    if key == ord('s'):
        with open('canny_parametri.txt', 'w') as f:
            f.write(f'Blur kernel: {k}\n')
            f.write(f'Sigma: {sigma:.2f}\n')
            f.write(f'Threshold1: {t1}\n')
            f.write(f'Threshold2: {t2}\n')
        print('Parametri salvati in "canny_parametri.txt"')

    # ESC per uscire
    if key == 27:
        break

cv2.destroyAllWindows()

