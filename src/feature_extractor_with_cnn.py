"""
feature_extractor_with_cnn.py
────────────────────────────────────────────────────────────────────────────
Combines feature_extractor.py (EAR / MAR / blink / head-pose logging)
with the MRL-Eye CNN (PyTorch) from mrl_eye_cnn.ipynb.

Every 10 seconds one row is appended to the CSV:
    timestamp, EAR_min, EAR_var, MAR_mean, MAR_var,
    blink_count_10s, avg_blink_duration_10s, pitch, roll,
    cnn_eye_state, cnn_p_open

cnn_eye_state : "awake" | "sleepy"
cnn_p_open    : mean raw sleepy-probability over the 10s window
                0.0 = definitely awake, 1.0 = definitely sleepy
                (ImageFolder alphabetical: index 0=awake, index 1=sleepy)
"""

from FaceMeshModule import FaceMeshDetection

import cv2
import time
import math
import numpy as np
import csv
import os
from PIL import Image


try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not found – CNN columns will be 'no_model'.")

VIDEO_PATH     = "/media/rudhratej/linux_data/projects/personal/multimodal_driver_fatigue_monitor/data/Awake_11_laps_albert.mov"           

WINDOW_TIME        = 10
LOG_INTERVAL       = 10
BASELINE_WINDOW    = 25
BLINK_RATIO_THRESH = 0.75

BASE_DIR  = "data"
BASE_NAME = "feature_log"
EXT       = ".csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1   = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3   = nn.Conv2d(32, 64, 3, padding=1)
        self.relu    = nn.ReLU()
        self.pool    = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1     = nn.Linear(64 * 8 * 8, 128)  
        self.fc2     = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
]) if TORCH_AVAILABLE else None


def rotation_matrix_to_euler(R):
    roll  = math.atan2(R[1, 0], R[1, 1])
    pitch = math.atan2(-R[1, 2], math.sqrt(R[1, 0]**2 + R[1, 1]**2))
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
    return [(t, v) for (t, v) in history if now - t <= window]


cnn_model = None
if TORCH_AVAILABLE and os.path.exists(CNN_MODEL_PATH):
    try:
        cnn_model = CNN().to(DEVICE)
        cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=True))
        cnn_model.eval()
        print(f"[INFO] CNN loaded from {CNN_MODEL_PATH} on {DEVICE}")
    except Exception as e:
        print(f"[WARN] Could not load CNN: {e}")
elif TORCH_AVAILABLE:
    print(f"[WARN] No model at '{CNN_MODEL_PATH}' – cnn columns will be 'no_model'/-1")


LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def crop_eye(img, face, indices):
    
    pts = np.array([face[i] for i in indices], dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    pad_x = max(int(w * 0.2), 4)
    pad_y = max(int(h * 0.2), 4)
    x1 = max(x - pad_x, 0);        y1 = max(y - pad_y, 0)
    x2 = min(x + w + pad_x, img.shape[1])
    y2 = min(y + h + pad_y, img.shape[0])
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

def cnn_predict(img, face):
  
    if cnn_model is None:
        return "no_model", -1.0

    p_sleepy_vals = []
    for idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
        pil_crop = crop_eye(img, face, idx)
        if pil_crop is None:
            continue
        tensor = INFER_TRANSFORM(pil_crop).unsqueeze(0).to(DEVICE)  # (1,1,64,64), normalised
        with torch.no_grad():
            probs = torch.softmax(cnn_model(tensor), dim=1)
            p_sleepy_vals.append(probs[0, 1].item())  # index 1 = sleepy

    if not p_sleepy_vals:
        return "unknown", -1.0

    mean_p_sleepy = float(np.mean(p_sleepy_vals))
    state = "sleepy" if mean_p_sleepy >= 0.5 else "awake"
    return state, round(mean_p_sleepy, 4)

detector = FaceMeshDetection()
cap      = cv2.VideoCapture(VIDEO_PATH)
ptime    = time.time()

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

blink_history        = []
blink_state          = 0
blink_start_time     = 0

ear_baseline_history = []
mar_baseline_history = []
ear_ratio_history    = []
mar_ratio_history    = []
cnn_window           = []   

REFERENCE_3D_FACE = np.array([
    (  0.0,   0.0,   0.0),
    (  0.0, -63.6, -12.0),
    (-45.0,  17.0, -20.0),
    ( 45.0,  17.0, -20.0),
    (-30.0, -50.0, -12.0),
    ( 30.0, -50.0, -12.0),
], dtype=np.float64)

CSV_LOG_PATH = get_next_log_path(BASE_DIR, BASE_NAME, EXT)
csv_file     = open(CSV_LOG_PATH, "w", newline="")
csv_writer   = csv.writer(csv_file)
csv_writer.writerow([
    "timestamp",
    "EAR_min", "EAR_var",
    "MAR_mean", "MAR_var",
    "blink_count_10s", "avg_blink_duration_10s",
    "pitch", "roll",
    "cnn_eye_state", "cnn_p_sleepy",
])

last_log_time = 0.0  
print(f"[INFO] Logging to {CSV_LOG_PATH}")
print("[INFO] Press 'q' to quit.")


cv2.namedWindow("capture", cv2.WINDOW_NORMAL)
while True:
    ctime = time.time()
    fps   = 1.0 / max(ctime - ptime, 1e-6)
    ptime = ctime

    ret, img = cap.read()
    if not ret:
        print("[INFO] End of video.")
        break

    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) 
    video_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  

    h, w = img.shape[:2]
    focal_length            = w
    camera_intrinsic_matrix = np.array(
        [[focal_length, 0, w/2],
         [0, focal_length, h/2],
         [0, 0, 1]], dtype=np.float64)
    distortion_coefficients = np.zeros((4, 1))

    img, faces = detector.findFaceMesh(img, False)
    p = r = y_angle = 0.0

    if faces:
        face = faces[0]

        
        left_eye_out_corner = face[33];  left_up_eye_in  = face[160]; left_up_eye_out  = face[158]
        left_eye_in_corner  = face[133]; left_low_eye_out= face[153]; left_low_eye_in  = face[144]
        right_eye_out_corner= face[362]; right_up_eye_in = face[385]; right_up_eye_out = face[387]
        right_eye_in_corner = face[263]; right_low_eye_out=face[373]; right_low_eye_in = face[380]

        EAR_left  = (abs_euclid_dist(left_up_eye_in,  left_low_eye_in)  + abs_euclid_dist(left_up_eye_out,  left_low_eye_out))  / (2 * abs_euclid_dist(left_eye_out_corner,  left_eye_in_corner))
        EAR_right = (abs_euclid_dist(right_up_eye_in, right_low_eye_in) + abs_euclid_dist(right_up_eye_out, right_low_eye_out)) / (2 * abs_euclid_dist(right_eye_out_corner, right_eye_in_corner))
        EAR_mean  = (EAR_left + EAR_right) / 2

        eye_points = [face[i] for i in LEFT_EYE_IDX + RIGHT_EYE_IDX]
        for pt in eye_points:
            cv2.circle(img, pt, 2, (255, 0, 255), -1)

        cv2.putText(img, f'EAR MEAN:{EAR_mean:.3f}', (20, 90),  cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(img, f'EAR LEFT:{EAR_left:.3f}',  (20, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(img, f'EAR RIGHT:{EAR_right:.3f}',(20, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

        
        mouth_left_corner  = face[61];  mouth_right_corner = face[291]
        mouth_left_up_lip  = face[78];  mouth_up_lip       = face[13]
        mouth_right_low_lip= face[308]; mouth_low_lip      = face[14]

        for pt in [face[61], face[291], face[78], face[13], face[308], face[14]]:
            cv2.circle(img, pt, 2, (255, 0, 255), -1)

        MAR = (abs_euclid_dist(mouth_up_lip, mouth_low_lip) + abs_euclid_dist(mouth_left_up_lip, mouth_right_low_lip)) / \
              (2 * abs_euclid_dist(mouth_left_corner, mouth_right_corner))

        cv2.putText(img, f'MAR:{MAR:.3f}',  (20, 180), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70),  cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

        
        now = video_ts
        ear_baseline_history = purge_old(ear_baseline_history, now, BASELINE_WINDOW)
        mar_baseline_history = purge_old(mar_baseline_history, now, BASELINE_WINDOW)

        ear_baseline = np.mean([v for _, v in ear_baseline_history]) if len(ear_baseline_history) > 5 else EAR_mean
        mar_baseline = np.mean([v for _, v in mar_baseline_history]) if len(mar_baseline_history) > 5 else MAR

        EAR_ratio = EAR_mean / (ear_baseline + 1e-6)
        MAR_ratio = MAR      / (mar_baseline + 1e-6)

        if EAR_ratio > 0.85:
            ear_baseline_history.append((now, EAR_mean))
        if MAR < 0.6:
            mar_baseline_history.append((now, MAR))

        ear_ratio_history.append((now, EAR_ratio))
        mar_ratio_history.append((now, MAR_ratio))

        now = video_ts
        if EAR_ratio < BLINK_RATIO_THRESH and blink_state == 0:
            blink_state      = 1
            blink_start_time = now
        elif EAR_ratio >= BLINK_RATIO_THRESH and blink_state == 1:
            blink_state = 0
            duration    = now - blink_start_time
            blink_history.append((now, duration))

        
        cnn_state, cnn_conf = cnn_predict(img, face)
        cnn_window.append((now, cnn_state, cnn_conf))
        cv2.putText(img, f"CNN:{cnn_state} p={cnn_conf:.3f}", (20, 210),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 200, 255), 2)

        
        now = video_ts
        ear_ratio_history = purge_old(ear_ratio_history, now, WINDOW_TIME)
        mar_ratio_history = purge_old(mar_ratio_history, now, WINDOW_TIME)
        blink_history     = purge_old(blink_history,     now, WINDOW_TIME)
        cnn_window        = [(t, s, c) for t, s, c in cnn_window if now - t <= WINDOW_TIME]

        
        face_2d_points = np.array([face[1], face[152], face[33],
                                   face[263], face[61], face[291]], dtype=np.float64)
        success, rotation_vector, _ = cv2.solvePnP(
            REFERENCE_3D_FACE, face_2d_points,
            camera_intrinsic_matrix, distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            R_cw, _ = cv2.Rodrigues(rotation_vector)
            R_wc    = R_cw.T
            p, y_angle, r = rotation_matrix_to_euler(R_wc)

        cv2.putText(img, f"pitch:{p:.2f}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"yaw:{y_angle:.2f}",   (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"roll:{r:.2f}",  (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        
        ear_values  = [v for _, v in ear_ratio_history]
        mar_values  = [v for _, v in mar_ratio_history]
        blink_times = [v for _, v in blink_history]

        blink_count_10s        = len(blink_times)
        avg_blink_duration_10s = sum(blink_times) / blink_count_10s if blink_count_10s > 0 else 0
        EAR_min  = min(ear_values)            if ear_values       else 0
        EAR_var  = float(np.var(ear_values))  if len(ear_values) > 1 else 0
        MAR_mean = float(np.mean(mar_values)) if mar_values       else 0
        MAR_var  = float(np.var(mar_values))  if len(mar_values) > 1 else 0

        valid_scores = [c for _, _, c in cnn_window if c >= 0.0]
        if valid_scores:
            cnn_mean_score = float(np.mean(valid_scores))
            cnn_majority   = "sleepy" if cnn_mean_score >= 0.5 else "awake"
        else:
            cnn_mean_score = cnn_conf
            cnn_majority   = cnn_state

        
        now = video_ts
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
                  f"blinks={blink_count_10s}  cnn={cnn_majority}  p_sleepy={cnn_mean_score:.4f}")

    else:
        print("[INFO] No face detected.")

    cv2.imshow("capture", img)
    cv2.resizeWindow("capture", 400, 700)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
csv_file.close()
print(f"[DONE] Log saved to {CSV_LOG_PATH}")