"""
feature_extractor_with_cnn.py
────────────────────────────────────────────────────────────────────────────
Combines feature_extractor.py (EAR / MAR / blink / head-pose logging)
with the MRL-Eye CNN from mrl_eye_cnn.ipynb.

Every 10 seconds one row is appended to the CSV:
    timestamp, EAR_min, EAR_var, MAR_mean, MAR_var,
    blink_count_10s, avg_blink_duration_10s, pitch, roll,
    cnn_eye_state, cnn_p_open

cnn_eye_state : "open" | "closed"  (derived from mean p_open over the window)
cnn_p_open    : mean raw open-probability over the 10-second window
                0.0 = definitely closed, 1.0 = definitely open

Requirements
    pip install opencv-python mediapipe numpy tensorflow

Put your trained model at:  models/mrl_eye_cnn.h5
  OR set  CNN_MODEL_PATH  to the correct path below.

If no model file is found the script still runs; cnn_eye_state will be "no_model".
"""

from FaceMeshModule import FaceMeshDetection   # your existing module

import cv2
import mediapipe as mp
import time
import math
import numpy as np
import csv
import os

# ── try loading TensorFlow / Keras ──────────────────────────────────────────
try:
    from tensorflow.keras.models import load_model as keras_load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not found – CNN eye state will be 'no_model'.")

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────
CNN_MODEL_PATH   = "models/mrl_eye_cnn.h5"   # path to your saved .h5 / SavedModel
CNN_INPUT_SIZE   = (24, 24)                   # (H, W) the model was trained on
                                               # change to (32,32) if needed
WINDOW_TIME      = 10   # seconds
LOG_INTERVAL     = 10   # seconds
BASELINE_WINDOW  = 25   # seconds
BLINK_RATIO_THRESH = 0.75

BASE_DIR  = "data"
BASE_NAME = "test_feature_log"
EXT       = ".csv"

# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────

def rotation_matrix_to_euler(R):
    roll  = math.atan2(R[1, 0], R[1, 1])
    pitch = math.atan2(-R[1, 2], math.sqrt(R[1, 0] ** 2 + R[1, 1] ** 2))
    yaw   = math.atan2(R[2, 1], R[2, 2])
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def abs_euclid_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_next_log_path(base_dir, base_name, ext):
    os.makedirs(base_dir, exist_ok=True)
    n = 1
    while True:
        path = os.path.join(base_dir, f"{base_name}_{n}{ext}")
        if not os.path.exists(path):
            return path
        n += 1


def purge_old(history, now, window):
    """Keep only entries within `window` seconds of `now`."""
    return [(t, v) for (t, v) in history if now - t <= window]


# ────────────────────────────────────────────────────────────────────────────
# CNN SETUP
# ────────────────────────────────────────────────────────────────────────────

cnn_model = None
if TF_AVAILABLE and os.path.exists(CNN_MODEL_PATH):
    try:
        cnn_model = keras_load_model(CNN_MODEL_PATH)
        print(f"[INFO] CNN model loaded from {CNN_MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Could not load CNN model: {e}")
elif TF_AVAILABLE:
    print(f"[WARN] No model found at '{CNN_MODEL_PATH}'. "
          "Place your mrl_eye_cnn.h5 there to enable CNN predictions.")


def crop_eye_region(img, landmarks, eye_indices):
    """
    Crop and resize a bounding-box region around the eye landmarks.
    landmarks : list of (x, y) pixel tuples (already pixel-space from FaceMesh)
    eye_indices: list of mediapipe landmark indices for one eye
    Returns a (1, H, W, 1) float32 array ready for the CNN, or None on failure.
    """
    pts = np.array([landmarks[i] for i in eye_indices], dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    # add 20% padding
    pad_x, pad_y = max(int(w * 0.2), 4), max(int(h * 0.2), 4)
    x1 = max(x - pad_x, 0);  y1 = max(y - pad_y, 0)
    x2 = min(x + w + pad_x, img.shape[1]);  y2 = min(y + h + pad_y, img.shape[0])
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    resized = cv2.resize(gray, (CNN_INPUT_SIZE[1], CNN_INPUT_SIZE[0]))
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, CNN_INPUT_SIZE[0], CNN_INPUT_SIZE[1], 1)


def cnn_predict_eye_state(img, face):
    """
    Run CNN on both eyes.
    Returns (state, p_open) where:
      state  : 'open' | 'closed' | 'unknown' | 'no_model'
      p_open : raw mean open-probability across both eyes, 0.0 → 1.0
               (0.0 = definitely closed, 1.0 = definitely open)
    """
    if cnn_model is None:
        return "no_model", -1.0

    LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    p_open_values = []
    for eye_idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
        crop = crop_eye_region(img, face, eye_idx)
        if crop is None:
            continue
        pred = cnn_model.predict(crop, verbose=0)[0]  # shape (2,) or (1,)
        if pred.shape[0] == 2:
            # output is [p_closed, p_open]  (MRL CNN convention)
            p_open = float(pred[1])
        else:
            # single sigmoid output: 1 = open
            p_open = float(pred[0])
        p_open_values.append(p_open)

    if not p_open_values:
        return "unknown", -1.0

    mean_p_open = float(np.mean(p_open_values))
    state = "open" if mean_p_open >= 0.5 else "closed"
    return state, round(mean_p_open, 4)


# ────────────────────────────────────────────────────────────────────────────
# MAIN LOOP STATE
# ────────────────────────────────────────────────────────────────────────────

detector = FaceMeshDetection()
cap      = cv2.VideoCapture(0)
ptime    = time.time()

blink_history      = []   # list of (end_time, duration)
blink_state        = 0
blink_start_time   = 0

ear_baseline_history = []
mar_baseline_history = []
ear_ratio_history    = []
mar_ratio_history    = []

# CNN window: list of (timestamp, 'open'|'closed', confidence)
cnn_window = []

# 3-D reference face model for PnP (mm)
REFERENCE_3D_FACE = np.array([
    (  0.0,   0.0,   0.0),   # nose tip
    (  0.0, -63.6, -12.0),   # chin
    (-45.0,  17.0, -20.0),   # left eye outer
    ( 45.0,  17.0, -20.0),   # right eye outer
    (-30.0, -50.0, -12.0),   # left mouth corner
    ( 30.0, -50.0, -12.0),   # right mouth corner
], dtype=np.float64)

# CSV setup
CSV_LOG_PATH = get_next_log_path(BASE_DIR, BASE_NAME, EXT)
csv_file     = open(CSV_LOG_PATH, "w", newline="")
csv_writer   = csv.writer(csv_file)
csv_writer.writerow([
    "timestamp",
    "EAR_min", "EAR_var",
    "MAR_mean", "MAR_var",
    "blink_count_10s", "avg_blink_duration_10s",
    "pitch", "roll",
    "cnn_eye_state", "cnn_p_open",
])

last_log_time = time.time()

print(f"[INFO] Logging to {CSV_LOG_PATH}")
print("[INFO] Press 'q' to quit.")

# ────────────────────────────────────────────────────────────────────────────
# CAPTURE LOOP
# ────────────────────────────────────────────────────────────────────────────

while True:
    ctime = time.time()
    fps   = 1.0 / max(ctime - ptime, 1e-6)
    ptime = ctime

    ret, img = cap.read()
    if not ret:
        print("[WARN] Frame capture failed – retrying.")
        continue

    h, w = img.shape[:2]
    focal_length            = w
    camera_intrinsic_matrix = np.array(
        [[focal_length, 0, w / 2],
         [0, focal_length, h / 2],
         [0, 0, 1]], dtype=np.float64)
    distortion_coefficients = np.zeros((4, 1))

    img, faces = detector.findFaceMesh(img, False)

    p = r = y_angle = 0.0   # head-pose defaults

    if faces:
        face = faces[0]

        # ── EAR ─────────────────────────────────────────────────────────────
        le_oc = face[33];  le_ui = face[160]; le_uo = face[158]
        le_ic = face[133]; le_lo = face[153]; le_li = face[144]
        re_oc = face[362]; re_ui = face[385]; re_uo = face[387]
        re_ic = face[263]; re_lo = face[373]; re_li = face[380]

        EAR_left  = (abs_euclid_dist(le_ui, le_li) + abs_euclid_dist(le_uo, le_lo)) / \
                    (2 * abs_euclid_dist(le_oc, le_ic))
        EAR_right = (abs_euclid_dist(re_ui, re_li) + abs_euclid_dist(re_uo, re_lo)) / \
                    (2 * abs_euclid_dist(re_oc, re_ic))
        EAR_mean  = (EAR_left + EAR_right) / 2

        for pt in [face[33], face[160], face[158], face[133], face[153], face[144],
                   face[362], face[385], face[387], face[263], face[373], face[380]]:
            cv2.circle(img, pt, 2, (255, 0, 255), -1)

        cv2.putText(img, f"EAR:{EAR_mean:.3f}", (20, 90),  cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)

        # ── MAR ─────────────────────────────────────────────────────────────
        m_lc = face[61];  m_rc = face[291]
        m_lu = face[78];  m_u  = face[13]
        m_rl = face[308]; m_l  = face[14]

        MAR = (abs_euclid_dist(m_u, m_l) + abs_euclid_dist(m_lu, m_rl)) / \
              (2 * abs_euclid_dist(m_lc, m_rc))

        cv2.putText(img, f"MAR:{MAR:.3f}", (20, 120), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(img, f"FPS:{fps:.0f}",  (20,  60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)

        # ── Baseline normalisation ───────────────────────────────────────────
        now = time.time()
        ear_baseline_history = purge_old(ear_baseline_history, now, BASELINE_WINDOW)
        mar_baseline_history = purge_old(mar_baseline_history, now, BASELINE_WINDOW)

        ear_baseline = (np.mean([v for _, v in ear_baseline_history])
                        if len(ear_baseline_history) > 5 else EAR_mean)
        mar_baseline = (np.mean([v for _, v in mar_baseline_history])
                        if len(mar_baseline_history) > 5 else MAR)

        EAR_ratio = EAR_mean / (ear_baseline + 1e-6)
        MAR_ratio = MAR      / (mar_baseline + 1e-6)

        if EAR_ratio > 0.85:
            ear_baseline_history.append((now, EAR_mean))
        if MAR < 0.6:
            mar_baseline_history.append((now, MAR))

        ear_ratio_history.append((now, EAR_ratio))
        mar_ratio_history.append((now, MAR_ratio))

        # ── Blink detection ──────────────────────────────────────────────────
        now = time.time()
        if EAR_ratio < BLINK_RATIO_THRESH and blink_state == 0:
            blink_state      = 1
            blink_start_time = now
        elif EAR_ratio >= BLINK_RATIO_THRESH and blink_state == 1:
            blink_state    = 0
            blink_end_time = now
            duration       = blink_end_time - blink_start_time
            blink_history.append((blink_end_time, duration))

        # ── CNN eye state ────────────────────────────────────────────────────
        cnn_state, cnn_conf = cnn_predict_eye_state(img, face)
        cnn_window.append((now, cnn_state, cnn_conf))

        cv2.putText(img, f"CNN:{cnn_state} p={cnn_conf:.3f}", (20, 150),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 200, 255), 2)

        # ── Purge windows ────────────────────────────────────────────────────
        now = time.time()
        ear_ratio_history = purge_old(ear_ratio_history, now, WINDOW_TIME)
        mar_ratio_history = purge_old(mar_ratio_history, now, WINDOW_TIME)
        blink_history     = purge_old(blink_history,     now, WINDOW_TIME)
        cnn_window        = [(t, s, c) for (t, s, c) in cnn_window
                             if now - t <= WINDOW_TIME]

        # ── Head pose (PnP) ──────────────────────────────────────────────────
        face_2d = np.array([face[1], face[152], face[33],
                            face[263], face[61], face[291]], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            REFERENCE_3D_FACE, face_2d,
            camera_intrinsic_matrix, distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE)

        if success:
            R_cw, _ = cv2.Rodrigues(rvec)
            R_wc    = R_cw.T
            p, y_angle, r = rotation_matrix_to_euler(R_wc)

        cv2.putText(img, f"pitch:{p:.1f}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(img, f"yaw:{y_angle:.1f}",   (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(img, f"roll:{r:.1f}",  (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # ── Aggregate features ───────────────────────────────────────────────
        ear_values  = [v for _, v in ear_ratio_history]
        mar_values  = [v for _, v in mar_ratio_history]
        blink_times = [v for _, v in blink_history]

        blink_count_10s       = len(blink_times)
        avg_blink_duration_10s = (sum(blink_times) / blink_count_10s
                                   if blink_count_10s > 0 else 0.0)
        EAR_min  = min(ear_values) if ear_values else 0.0
        EAR_var  = float(np.var(ear_values))  if len(ear_values) > 1 else 0.0
        MAR_mean = float(np.mean(mar_values)) if mar_values else 0.0
        MAR_var  = float(np.var(mar_values))  if len(mar_values) > 1 else 0.0

        # CNN: mean p_open score over the 10-second window
        valid_scores = [score for _, _, score in cnn_window if score >= 0.0]
        if valid_scores:
            cnn_mean_score = float(np.mean(valid_scores))
            cnn_majority   = "open" if cnn_mean_score >= 0.5 else "closed"
        else:
            cnn_mean_score = cnn_conf   # last single frame (may be -1.0 if no model)
            cnn_majority   = cnn_state

        # ── CSV write every LOG_INTERVAL seconds ─────────────────────────────
        now = time.time()
        if now - last_log_time >= LOG_INTERVAL:
            csv_writer.writerow([
                round(now, 3),
                round(EAR_min,  4), round(EAR_var,  4),
                round(MAR_mean, 4), round(MAR_var,  4),
                blink_count_10s, round(avg_blink_duration_10s, 4),
                round(p, 3), round(r, 3),
                cnn_majority, round(cnn_mean_score, 4),
            ])
            csv_file.flush()
            last_log_time = now
            print(f"[LOG] t={now:.1f}  EAR_min={EAR_min:.3f}  "
                  f"blinks={blink_count_10s}  cnn={cnn_majority}  p_open={cnn_mean_score:.4f}")

    else:
        print("[INFO] No face detected – waiting.")

    cv2.imshow("Driver Fatigue Monitor", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ── Cleanup ──────────────────────────────────────────────────────────────────
cv2.destroyAllWindows()
cap.release()
csv_file.close()
print(f"[DONE] Log saved to {CSV_LOG_PATH}")
