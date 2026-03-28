from pseyepy import Camera
import cv2
import numpy as np
import json
import os

CHECKERBOARD = (3, 4)  # inner corners
SQUARE_SIZE = 0.03    # meters - measure your printed square size and update this

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

cam = Camera(0, fps=60, resolution=Camera.RES_LARGE, colour=True, exposure=10000)

print("Hold checkerboard in front of camera.")
print("Press SPACE to capture, need ~20 good captures. Press Q to finish and calibrate.")
print("Use trackbars to adjust contrast/brightness for better board detection.")

# --- Trackbar setup ---
WINDOW = 'Calibration'
cv2.namedWindow(WINDOW)
cv2.createTrackbar('Contrast',   WINDOW, 10, 30, lambda x: None)  # value / 10 → alpha
cv2.createTrackbar('Brightness', WINDOW, 0, 100, lambda x: None)  # direct beta offset
cv2.createTrackbar('Exposure',   WINDOW, 100, 200, lambda x: None) # relative ×100

def apply_contrast_brightness(frame):
    """alpha = contrast (1.0=neutral), beta = brightness offset."""
    alpha = cv2.getTrackbarPos('Contrast',   WINDOW) / 10.0  # 0.0–3.0, default 1.0
    beta  = cv2.getTrackbarPos('Brightness', WINDOW) - 50    # -50 to +50, default 0
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

count = 0
prev_exposure_val = None

while True:
    frame, _ = cam.read()

    # Live exposure update via trackbar (maps 0–200 → 0–20000)
    exposure_val = cv2.getTrackbarPos('Exposure', WINDOW) * 100
    if exposure_val != prev_exposure_val:
        cam.exposure = exposure_val
        prev_exposure_val = exposure_val

    display = apply_contrast_brightness(frame)
    gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners, ret)
        cv2.putText(display, 'Board found! SPACE to capture', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display, 'Board not found', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    alpha = cv2.getTrackbarPos('Contrast', WINDOW) / 10.0
    beta  = cv2.getTrackbarPos('Brightness', WINDOW) - 50
    cv2.putText(display, f'Captures: {count}  |  a={alpha:.1f}  b={beta:+d}  exp={exposure_val}',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow(WINDOW, display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') and ret:
        # Always calibrate on the raw (unprocessed) frame for accuracy
        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners2 = cv2.cornerSubPix(gray_raw, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        count += 1
        print(f'Captured {count}')
    elif key == ord('q') and count >= 10:
        break

cam.end()
cv2.destroyAllWindows()

print('Calibrating...')
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(f'Reprojection error: {ret:.4f} (lower is better, aim for <0.5)')

params = {
    "intrinsic_matrix": K.tolist(),
    "distortion_coefficients": dist.tolist()
}

os.makedirs('../public', exist_ok=True)
with open('../public/camera_params.json', 'w') as f:
    json.dump({"cam0": params}, f, indent=2)

print('Saved to public/camera_params.json')
print('Intrinsic matrix:')
print(K)