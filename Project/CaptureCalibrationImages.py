import cv2 as cv
import os

CHESS_BOARD_DIM = (4, 3)

n = 0  # image counter

# checking if images dir exists, if not then create it
image_dir_path = "calib_images"
os.makedirs(image_dir_path, exist_ok=True)

# criteria for chessboard detection
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv.findChessboardCorners(grayImage, boardDimension)
    if ret:
        corners1 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)
    return image, ret

cap = cv.VideoCapture(0)

# create resizable windows
cv.namedWindow("frame",     cv.WINDOW_NORMAL)
cv.namedWindow("copyFrame", cv.WINDOW_NORMAL)
cv.setWindowProperty("frame", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
cv.setWindowProperty("copyFrame", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)


# set initial size (adjust to your screen or camera resolution)
# cv.resizeWindow("frame",     2048, 1088)
# cv.resizeWindow("copyFrame", 2048, 1088)



while True:
    ret, frame = cap.read()
    if not ret:
        break

    copyFrame = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    image, board_detected = detect_checker_board(frame, gray, criteria, CHESS_BOARD_DIM)
    print(board_detected)

    cv.putText(
        frame,
        f"saved_img : {n}",
        (30, 40),
        cv.FONT_HERSHEY_PLAIN,
        1.4,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )
    cv.namedWindow("frame", cv.WINDOW_NORMAL)
    cv.resizeWindow("frame", 800, 600)
    cv.imshow("frame", frame)
    cv.imwrite(f"{image_dir_path}/image_frame{n}.png", frame)
    cv.waitKey(1000)
    cv.namedWindow(" copy frame", cv.WINDOW_NORMAL)
    cv.resizeWindow("copy frame", 800, 600)
    cv.imshow("copy frame", frame)
    cv.imwrite(f"{image_dir_path}/image_{n}.png", copyFrame)
    cv.imshow("copy frame", copyFrame)
    cv.waitKey(1000)

    key = cv.waitKey(1) & 0xFF 
    if key == ord("q"):
        break
    if key == ord("s") and board_detected:
        cv.imwrite(f"{image_dir_path}/image{n}.png", copyFrame)
        print(f"saved image number {n}")
        n += 1

cap.release()
cv.destroyAllWindows()

print("Total saved Images:", n)
