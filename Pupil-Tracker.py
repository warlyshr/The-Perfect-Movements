from __future__ import annotations

"""
Eye‑gesture control — lean, no‑speech build
==========================================

Tuned settings for potentially improved responsiveness and accuracy.
Debug window enabled by default for real-time feedback.

Features:
* Left and right eye movement detection
* Blink for clicking
* Long blink to pause/resume command execution
* Smoother tracking (less averaging lag)
* Faster reaction time (fewer stable frames required)
* Increased horizontal tolerance (higher TOLERANCE)
* Shorter command cooldown
"""

import sys, time, cv2, mediapipe as mp, numpy as np, pyautogui
from collections import deque
from typing import Dict

# ---------------- USER SETTINGS (Tuned) ----------------
CAM_INDEX = 0
CAM_WIDTH, CAM_HEIGHT = 640, 480
FPS_LIMIT = 30               # leaner default
CALIB_SEC_PER_POSE = 4.0
SMOOTH = 3                   # Reduced for less lag
STABLE_FRAMES = 2            # Reduced for faster reaction
MOVE_PIXELS = 100
CURSOR_DURATION = 0          # non‑blocking cursor nudge
COOLDOWN_SECONDS = 0.7       # Reduced for quicker commands
USE_ARROW_KEYS = False
# Horizontal classifier
TOLERANCE = {"left": 0.65, "right": 0.65} # Increased tolerance if left and right misses increase them
# Blink
BLINK_THRESHOLD = 0.19
BLINK_COOLDOWN_S = 0.6
# Long blink for pause/resume
LONG_BLINK_THRESHOLD = 0.19  # Same as regular blink threshold
LONG_BLINK_DURATION = 1.0    # Seconds eyes must remain closed
PAUSE_RESUME_COOLDOWN = 2.0  # Prevent accidental toggling
# Debug
DEBUG_WINDOW = True          # Enabled by default
DOT_RADIUS = 45
# ------------------------------------------------

# ---------- camera & FaceMesh -------------------
cam = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
if not cam.isOpened():
    sys.exit(f"❌ Cannot open camera {CAM_INDEX}")

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

SCREEN_W, SCREEN_H = pyautogui.size()
pyautogui.moveTo(SCREEN_W // 2, SCREEN_H // 2)

# Landmarks
L_EYE, R_EYE = [33, 133], [362, 263]
L_IRIS, R_IRIS = 468, 473
L_TOP, L_BOTTOM = 159, 145
R_TOP, R_BOTTOM = 386, 374
np_norm = np.linalg.norm

def to_px(lm, idx):
    p = lm[idx]
    return np.array([int(p.x * CAM_WIDTH), int(p.y * CAM_HEIGHT)])

def iris_ratio(lm):
    li, ri = to_px(lm, L_IRIS), to_px(lm, R_IRIS)
    lL, lR = to_px(lm, L_EYE[0]), to_px(lm, L_EYE[1])
    rL, rR = to_px(lm, R_EYE[0]), to_px(lm, R_EYE[1])
    # Handle potential division by zero if eye width is zero
    l_eye_width = lR[0] - lL[0]
    r_eye_width = rR[0] - rL[0]
    rx_l = (li[0] - lL[0]) / (l_eye_width + 1e-6) if l_eye_width != 0 else 0
    rx_r = (ri[0] - rL[0]) / (r_eye_width + 1e-6) if r_eye_width != 0 else 0
    rx = (rx_l + rx_r) / 2

    lT, lB = to_px(lm, L_TOP), to_px(lm, L_BOTTOM)
    rT, rB = to_px(lm, R_TOP), to_px(lm, R_BOTTOM)
    # Handle potential division by zero if eye height is zero
    l_eye_height = lB[1] - lT[1]
    r_eye_height = rB[1] - rT[1]
    ry_l = (li[1] - lT[1]) / (l_eye_height + 1e-6) if l_eye_height != 0 else 0
    ry_r = (ri[1] - rT[1]) / (r_eye_height + 1e-6) if r_eye_height != 0 else 0
    ry = (ry_l + ry_r) / 2

    return np.array([rx, ry])

def ear(lm):
    lT, lB = to_px(lm, L_TOP), to_px(lm, L_BOTTOM)
    rT, rB = to_px(lm, R_TOP), to_px(lm, R_BOTTOM)
    lL, lR = to_px(lm, L_EYE[0]), to_px(lm, L_EYE[1])
    rL, rR = to_px(lm, R_EYE[0]), to_px(lm, R_EYE[1])
    def _ear(T, B, L, R):
        eye_width = np_norm(L - R)
        # Avoid division by zero
        return np_norm(T - B) / (eye_width + 1e-6) if eye_width != 0 else 0
    ear_l = _ear(lT, lB, lL, lR)
    ear_r = _ear(rT, rB, rL, rR)
    return (ear_l + ear_r) / 2

# ---------- calibration -------------------------
CALIB_WIN = "Calibrate"
cv2.namedWindow(CALIB_WIN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(CALIB_WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
blank = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

def dot(pt):
    fr = blank.copy()
    cv2.circle(fr, pt, DOT_RADIUS, (255, 255, 255), -1)
    cv2.imshow(CALIB_WIN, fr)

# Removed up and down poses, keeping only left, right, and center
POSES = [
    ("left", (SCREEN_W//5, SCREEN_H//2)),
    ("right", (SCREEN_W*4//5, SCREEN_H//2)),
    ("centre", (SCREEN_W//2, SCREEN_H//2)),
]
centroids: Dict[str, np.ndarray] = {}
print("🧭 Calibration – follow the dot")
for k, pt in POSES:
    print(f"   → look {k}")
    buf = []
    end = time.time() + CALIB_SEC_PER_POSE
    while time.time() < end:
        dot(pt)
        ok, frm = cam.read()
        if not ok:
            print("Warning: Camera frame read failed during calibration.")
            time.sleep(0.1) # Avoid busy-looping if camera fails
            continue
        # Ensure frame is not empty
        if frm is None or frm.size == 0:
            print("Warning: Empty camera frame received during calibration.")
            time.sleep(0.1)
            continue

        try:
            rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                # Add safety check for landmarks
                if len(lm) > max(L_IRIS, R_IRIS, L_TOP, L_BOTTOM, R_TOP, R_BOTTOM, L_EYE[0], L_EYE[1], R_EYE[0], R_EYE[1]):
                     buf.append(iris_ratio(lm))
                else:
                    print("Warning: Not enough landmarks detected during calibration.")
            else:
                 print("Warning: No face landmarks detected during calibration frame.")
        except Exception as e:
             print(f"Error processing frame during calibration: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
            sys.exit("Calibration aborted by user.")

    if not buf:
         # Handle case where no valid data was collected for a pose
         print(f"❌ ERROR: No valid eye data collected for '{k}' pose. Calibration failed.")
         print("Ensure good lighting and clear view of the face.")
         cam.release()
         cv2.destroyAllWindows()
         sys.exit("Calibration failed.")
    centroids[k] = np.mean(buf, axis=0)

cv2.destroyWindow(CALIB_WIN)

# Check if all required centroids were captured
required_keys = {"left", "right", "centre"}
if not required_keys.issubset(centroids.keys()):
    print("❌ ERROR: Calibration did not capture all required poses.")
    cam.release()
    cv2.destroyAllWindows()
    sys.exit("Incomplete calibration.")

centre = centroids['centre']
# Calculate offsets and thresholds for left-right only
base_dist = {k: np_norm(v-centre) for k, v in centroids.items() if k in ('left', 'right')}

print(f"Centre: {centre}")
print(f"Base distances - Left: {base_dist.get('left', 0):.3f}, Right: {base_dist.get('right', 0):.3f}")
print("✅ Calibration complete")

# ---------- main loop ---------------------------
sm_x, sm_y = deque(maxlen=SMOOTH), deque(maxlen=SMOOTH)
last_emit = 0.0
stable = 0
prev = 'centre'
blink_cd = 0.0
prev_t = time.time()

# Pause/resume variables
commands_paused = False
eye_close_start = 0
last_pause_toggle = 0

while True:
    ok, frm = cam.read()
    if not ok:
        print("Warning: Camera frame read failed in main loop.")
        time.sleep(0.1) # Avoid busy-looping
        continue
    # Ensure frame is not empty
    if frm is None or frm.size == 0:
        print("Warning: Empty camera frame received in main loop.")
        time.sleep(0.1)
        continue

    try:
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
    except Exception as e:
        print(f"Error processing frame in main loop: {e}")
        continue # Skip this frame

    now = time.time()
    label = 'centre'
    med = np.array([np.median(sm_x) if sm_x else centre[0], np.median(sm_y) if sm_y else centre[1]]) # Default to centre if deque is empty
    current_ear = 0  # Default value for current_ear

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        # Add safety check for landmarks
        if len(lm) > max(L_IRIS, R_IRIS, L_TOP, L_BOTTOM, R_TOP, R_BOTTOM, L_EYE[0], L_EYE[1], R_EYE[0], R_EYE[1]):
            vec = iris_ratio(lm)
            sm_x.append(vec[0]); sm_y.append(vec[1])
            med = np.array([np.median(sm_x), np.median(sm_y)])

            # horizontal classification only
            dists = {k: np_norm(med - c) for k, c in centroids.items() if k in ('left', 'right')}
            if dists: # Ensure we have distances to compare
                 best = min(dists, key=dists.get)
                 # Check if base_dist for the best direction exists and is non-zero
                 if best in base_dist and base_dist[best] > 1e-6:
                     if dists[best] <= TOLERANCE[best] * base_dist[best]:
                         label = best
                 else:
                      # Fallback or warning if base distance is missing or zero
                      pass # Keep label as 'centre' or previous logic

            # Eye aspect ratio calculation
            current_ear = ear(lm)
            
            # Regular blink detection
            if current_ear < BLINK_THRESHOLD and now > blink_cd:
                print(f"👁️ Blink detected (EAR: {current_ear:.3f})")
                pyautogui.click()
                blink_cd = now + BLINK_COOLDOWN_S
                last_emit = now # Prevent move immediately after click
                stable = 0 # Reset stability after click
            
            # Long blink detection for pause/resume
            if current_ear < LONG_BLINK_THRESHOLD:
                if eye_close_start == 0:  # Eyes just closed
                    eye_close_start = now
                elif now - eye_close_start >= LONG_BLINK_DURATION and now > last_pause_toggle + PAUSE_RESUME_COOLDOWN:
                    # Long blink detected - toggle pause state
                    commands_paused = not commands_paused
                    print(f"👁️ Commands {'PAUSED' if commands_paused else 'RESUMED'} (long blink: {now - eye_close_start:.2f}s)")
                    last_pause_toggle = now
                    eye_close_start = 0  # Reset timer
            elif current_ear >= LONG_BLINK_THRESHOLD:
                eye_close_start = 0  # Reset if eyes opened before threshold
                
        else:
            print("Warning: Not enough landmarks detected in main loop.")

    # stability filter
    if label == prev:
        stable += 1
    else:
        stable = 1 # Reset counter
        prev = label

    # emit command - only if not paused
    if not commands_paused and label != 'centre' and stable >= STABLE_FRAMES and now - last_emit > COOLDOWN_SECONDS:
        print('➡️', label.upper(), f"(stable: {stable})")
        if USE_ARROW_KEYS:
            pyautogui.press(label)
        else:
            dx = {'left': -MOVE_PIXELS, 'right': MOVE_PIXELS}.get(label, 0)
            # No vertical movement (dy is always 0)
            pyautogui.moveRel(dx, 0, duration=CURSOR_DURATION)
        last_emit = now
        stable = 0 # Reset stability after emitting command

    # Debug Window Update
    if DEBUG_WINDOW:
        dbg = frm.copy() # Work on a copy
        if res.multi_face_landmarks and len(res.multi_face_landmarks[0].landmark) > max(L_IRIS, R_IRIS, L_TOP, L_BOTTOM, R_TOP, R_BOTTOM, L_EYE[0], L_EYE[1], R_EYE[0], R_EYE[1]):
            lm = res.multi_face_landmarks[0].landmark
            for idx in L_EYE + R_EYE + [L_IRIS, R_IRIS, L_TOP, L_BOTTOM, R_TOP, R_BOTTOM]:
                  try:
                      cv2.circle(dbg, tuple(to_px(lm, idx)), 2, (0, 255, 0), -1)
                  except IndexError:
                      print(f"Warning: Landmark index {idx} out of bounds.")
        
        # Display info texts
        cv2.putText(dbg, f'Dir: {label} (Stable: {stable})', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(dbg, f'X: {med[0]:.3f} Y: {med[1]:.3f}', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(dbg, f'EAR: {current_ear:.3f}', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Add pause state indicator
        status_color = (0, 0, 255) if commands_paused else (0, 255, 0)
        status_text = "COMMANDS PAUSED" if commands_paused else "COMMANDS ACTIVE"
        cv2.putText(dbg, f'Status: {status_text}', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
        
        # Long blink progress indicator when eyes are closed
        if eye_close_start > 0:
            close_duration = now - eye_close_start
            progress = min(close_duration / LONG_BLINK_DURATION, 1.0) * 100
            cv2.putText(dbg, f'Long Blink: {progress:.0f}%', (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        cv2.putText(dbg, f'FPS: {1.0 / (now - prev_t + 1e-6):.1f}', (CAM_WIDTH - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow('Debug', dbg)

    # Frame rate limiting and exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Precise sleep calculation
    elapsed = time.time() - prev_t
    sleep_time = (1.0 / FPS_LIMIT) - elapsed if FPS_LIMIT > 0 else 0
    if sleep_time > 0:
        time.sleep(sleep_time)
    prev_t = time.time() # Update prev_t after potential sleep

cam.release()
cv2.destroyAllWindows()
print("ℹ️ Script finished.")
