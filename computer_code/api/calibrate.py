from pseyepy import Camera
import cv2
import numpy as np
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

CHECKERBOARD = (6, 6)
SQUARE_SIZE = 0.02     # meters

CAPTURE_DIR = 'captures'

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

DETECTION_FLAGS = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                   cv2.CALIB_CB_NORMALIZE_IMAGE +
                   cv2.CALIB_CB_FILTER_QUADS)

WINDOW = 'Calibration'
REVIEW_WINDOW = 'Review  [SPACE=keep  D=discard  LEFT=back  Q=done]'


def make_window(exp=255, contrast=37, brightness=0, blur=2, thresh=0):
    cv2.destroyAllWindows()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Contrast',   WINDOW, contrast,   100, lambda x: None)
    cv2.createTrackbar('Brightness', WINDOW, brightness, 200, lambda x: None)
    cv2.createTrackbar('Exposure',   WINDOW, exp,        255, lambda x: None)
    cv2.createTrackbar('Blur(med)',  WINDOW, blur,         10, lambda x: None)
    cv2.createTrackbar('Threshold',  WINDOW, thresh,      255, lambda x: None)


def apply_contrast_brightness(gray):
    alpha = cv2.getTrackbarPos('Contrast',   WINDOW) / 10.0
    beta  = cv2.getTrackbarPos('Brightness', WINDOW) - 50
    return cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)


def apply_median_blur(img):
    k = cv2.getTrackbarPos('Blur(med)', WINDOW)
    if k == 0:
        return img
    return cv2.medianBlur(img, k * 2 + 1)


def apply_threshold(img):
    t = cv2.getTrackbarPos('Threshold', WINDOW)
    if t == 0:
        return img
    _, out = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
    return out


def review_captures(capture_dir, total):
    """
    Step through saved overlay images one at a time.
    SPACE = keep, D = discard, LEFT arrow = go back, Q = accept rest and finish.
    Returns sorted list of kept 0-based indices into objpoints/imgpoints.
    """
    cv2.namedWindow(REVIEW_WINDOW, cv2.WINDOW_NORMAL)

    # start with all unreviewed; track explicit decisions
    kept = set()
    discarded = set()

    i = 0
    while i < total:
        path = os.path.join(capture_dir, f'cap_{i:04d}.png')
        img = cv2.imread(path)
        if img is None:
            i += 1
            continue

        if i in kept:
            status_str = 'KEPT'
            color = (0, 255, 0)
        elif i in discarded:
            status_str = 'DISCARDED'
            color = (0, 0, 255)
        else:
            status_str = 'unreviewed'
            color = (0, 255, 255)

        display = img.copy()
        cv2.putText(display, f'{i+1}/{total}  [{status_str}]', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow(REVIEW_WINDOW, display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord(' '):
            kept.add(i)
            discarded.discard(i)
            i += 1
        elif key == ord('d'):
            discarded.add(i)
            kept.discard(i)
            i += 1
        elif key in (81, 2, ord('a')):   # left arrow or 'a' — go back
            i = max(0, i - 1)
        elif key == ord('q'):
            # keep everything not explicitly discarded
            for j in range(i, total):
                if j not in discarded:
                    kept.add(j)
            break
        # any other key: redisplay same frame

    cv2.destroyWindow(REVIEW_WINDOW)
    return sorted(kept)


def _ransac_iteration(args):
    """Single RANSAC iteration — runs in a worker thread."""
    objpoints, imgpoints, img_shape, n, subset_size, inlier_thresh, seed = args
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, subset_size, replace=False)
    sub_obj = [objpoints[j] for j in idx]
    sub_img = [imgpoints[j] for j in idx]

    try:
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            sub_obj, sub_img, img_shape, None, None
        )
    except cv2.error:
        return None

    idx_list = idx.tolist()
    inliers = []
    for j in range(n):
        if j in idx_list:
            rv = rvecs[idx_list.index(j)]
            tv = tvecs[idx_list.index(j)]
        else:
            _, rv, tv = cv2.solvePnP(objpoints[j], imgpoints[j], K, dist)
        proj, _ = cv2.projectPoints(objpoints[j], rv, tv, K, dist)
        err = cv2.norm(imgpoints[j], proj.reshape(-1, 1, 2), cv2.NORM_L2) / len(proj)
        if err < inlier_thresh:
            inliers.append(j)

    return (len(inliers), ret, inliers)


def ransac_calibrate(objpoints, imgpoints, img_shape, n_iter=200, inlier_thresh=0.5):
    n = len(objpoints)
    subset_size = max(10, n // 2)
    n_workers = os.cpu_count()

    print(f'RANSAC: {n} captures, {n_iter} iterations, subset={subset_size}, '
          f'thresh={inlier_thresh}px, workers={n_workers}')

    args = [
        (objpoints, imgpoints, img_shape, n, subset_size, inlier_thresh, i)
        for i in range(n_iter)
    ]

    best_inliers = []
    best_error = float('inf')

    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for result in executor.map(_ransac_iteration, args, chunksize=10):
            done += 1
            print(f'\r  RANSAC {done}/{n_iter}  best_inliers={len(best_inliers)}  best_err={best_error:.4f}',
                  end='', flush=True)
            if result is None:
                continue
            n_inliers, ret, inliers = result
            if n_inliers > len(best_inliers) or (
                n_inliers == len(best_inliers) and ret < best_error
            ):
                best_inliers = inliers
                best_error = ret
    print()

    if len(best_inliers) < 10:
        print(f'RANSAC: only {len(best_inliers)} inliers, falling back to all captures.')
        best_inliers = list(range(n))

    print(f'RANSAC: kept {len(best_inliers)}/{n} captures.')

    final_obj = [objpoints[j] for j in best_inliers]
    final_img = [imgpoints[j] for j in best_inliers]
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        final_obj, final_img, img_shape, None, None
    )
    return ret, K, dist, rvecs, tvecs, best_inliers


if __name__ == '__main__':
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    # clear old captures from previous run
    for f in os.listdir(CAPTURE_DIR):
        if f.startswith('cap_') and f.endswith('.png'):
            os.remove(os.path.join(CAPTURE_DIR, f))

    make_window(exp=244, blur=2)

    current_exposure = 244
    cam = Camera(0, fps=90, resolution=Camera.RES_SMALL, colour=False, exposure=current_exposure)

    objpoints = []
    imgpoints = []

    print(f"Checkerboard: {CHECKERBOARD} inner corners  (cols={CHECKERBOARD[0]}, rows={CHECKERBOARD[1]})")
    print("Controls:")
    print("  E - apply exposure trackbar (restarts camera)")
    print(f"  Q - stop capture, review '{CAPTURE_DIR}/', then RANSAC calibrate (need >= 10)")

    def restart_cam_with_exposure(exposure_val):
        global cam, current_exposure
        exposure_val = max(0, min(255, int(exposure_val)))
        contrast   = cv2.getTrackbarPos('Contrast',   WINDOW)
        brightness = cv2.getTrackbarPos('Brightness', WINDOW)
        blur       = cv2.getTrackbarPos('Blur(med)',  WINDOW)
        thresh     = cv2.getTrackbarPos('Threshold',  WINDOW)
        cam.end()
        time.sleep(0.3)
        cam = Camera(0, fps=90, resolution=Camera.RES_SMALL, colour=False, exposure=exposure_val)
        current_exposure = exposure_val
        make_window(exp=exposure_val, contrast=contrast, brightness=brightness, blur=blur, thresh=thresh)
        print(f'Exposure set to {exposure_val}')

    count = 0
    gray_raw = None

    while True:
        frame, _ = cam.read()

        if frame.ndim == 3:
            gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_raw = frame.copy()

        display = apply_contrast_brightness(gray_raw)
        detect  = apply_median_blur(display)
        detect  = apply_threshold(detect)
        found, corners = cv2.findChessboardCorners(detect, CHECKERBOARD, DETECTION_FLAGS)

        display_bgr = cv2.cvtColor(detect, cv2.COLOR_GRAY2BGR)

        if found:
            corners2 = cv2.cornerSubPix(gray_raw, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)

            # save overlay image to disk
            overlay = display_bgr.copy()
            cv2.drawChessboardCorners(overlay, CHECKERBOARD, corners, found)
            cv2.putText(overlay, f'cap {count:04d}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(CAPTURE_DIR, f'cap_{count:04d}.png'), overlay)

            count += 1

        if found:
            cv2.drawChessboardCorners(display_bgr, CHECKERBOARD, corners, found)
            cv2.putText(display_bgr, f'Capturing  [{count} frames]', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_bgr, f'Board not found  ({CHECKERBOARD[0]}x{CHECKERBOARD[1]} inner corners)',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        alpha       = cv2.getTrackbarPos('Contrast',   WINDOW) / 10.0
        beta        = cv2.getTrackbarPos('Brightness', WINDOW) - 50
        exp_pending = cv2.getTrackbarPos('Exposure',   WINDOW)
        blur_k      = cv2.getTrackbarPos('Blur(med)',  WINDOW)
        thresh_val  = cv2.getTrackbarPos('Threshold',  WINDOW)
        kernel_sz   = blur_k * 2 + 1 if blur_k > 0 else 0

        cv2.putText(display_bgr,
                    f'Captures: {count}  |  a={alpha:.1f}  b={beta:+d}  '
                    f'exp={current_exposure} (pending:{exp_pending})  blur={kernel_sz}px  thresh={thresh_val}  [E to apply exp]',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.imshow(WINDOW, display_bgr)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('e'):
            new_exposure = cv2.getTrackbarPos('Exposure', WINDOW)
            restart_cam_with_exposure(new_exposure)

        elif key == ord('q'):
            if count >= 10:
                break
            else:
                print(f'Need at least 10 captures, only have {count}.')

    cam.end()
    cv2.destroyAllWindows()

    if not objpoints:
        print('No captures, exiting.')
        exit(1)

    # --- manual review ---
    print(f'\n{count} captures saved to "{CAPTURE_DIR}/". Launching review...')
    print('SPACE=keep  D=discard  LEFT/A=go back  Q=accept remaining and finish')
    kept_indices = review_captures(CAPTURE_DIR, count)

    if len(kept_indices) < 10:
        print(f'Only {len(kept_indices)} kept after review (need >= 10). Exiting.')
        exit(1)

    print(f'Kept {len(kept_indices)}/{count} captures after review.')
    objpoints = [objpoints[i] for i in kept_indices]
    imgpoints = [imgpoints[i] for i in kept_indices]

    # --- RANSAC calibrate ---
    print(f'\nCalibrating with RANSAC over {len(objpoints)} captures...')
    ret, K, dist, rvecs, tvecs, inliers = ransac_calibrate(
        objpoints, imgpoints, gray_raw.shape[::-1]
    )

    print(f'Reprojection error: {ret:.4f} (lower is better, aim for <0.5)')
    print(f'Used {len(inliers)}/{len(objpoints)} captures after RANSAC.')

    params = {
        "intrinsic_matrix": K.tolist(),
        "distortion_coefficients": dist.tolist()
    }

    os.makedirs('../public', exist_ok=True)
    with open('../public/camera_params.json', 'w') as f:
        json.dump({"cam0": params}, f, indent=2)

    print('Saved to ../public/camera_params.json')
    print('Intrinsic matrix:')
    print(K)

    # clear captures folder
    for f in os.listdir(CAPTURE_DIR):
        if f.startswith('cap_') and f.endswith('.png'):
            os.remove(os.path.join(CAPTURE_DIR, f))
    print(f'Cleared "{CAPTURE_DIR}/".')