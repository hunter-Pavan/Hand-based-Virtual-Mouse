import time
from collections import deque

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ------------------ Config ------------------
SMOOTHING_ALPHA = 0.35
PINCH_THRESH = 0.035
SCROLL_PAIR_THRESH = 0.04
DRAG_HOLD_SECONDS = 0.5
CLICK_TAP_MAX_SECONDS = 0.35
ACTIVE_MARGIN = 0.12
CAM_INDEX = 0
SHOW_HUD = True
# --------------------------------------------

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

model_path = "hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# ---------- Utility ----------
def norm_dist(a, b):
    return np.linalg.norm(
        np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    )

class EMA:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None

    def update(self, new):
        new = np.array(new, dtype=float)
        if self.value is None:
            self.value = new
        else:
            self.value = self.alpha * new + (1 - self.alpha) * self.value
        return self.value

class GestureState:
    def __init__(self):
        self.pinching_index = False
        self.pinching_middle = False
        self.pinch_start_time = None
        self.dragging = False
        self.scroll_mode = False
        self.scroll_buffer = deque(maxlen=5)

    def start_pinch(self):
        self.pinch_start_time = time.time()

    def pinch_duration(self):
        return 0 if self.pinch_start_time is None else time.time() - self.pinch_start_time

def map_to_screen(nx, ny):
    m = ACTIVE_MARGIN
    nx = np.clip((nx - m) / (1 - 2 * m), 0, 1)
    ny = np.clip((ny - m) / (1 - 2 * m), 0, 1)
    nx = 1 - nx
    return int(nx * screen_w), int(ny * screen_h)

# ---------- MediaPipe ----------
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 24)

ema = EMA(SMOOTHING_ALPHA)
state = GestureState()
prev_x, prev_y = 0, 0
frame_id = 0

# ---------- Main Loop ----------
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        result = landmarker.detect_for_video(mp_image, frame_id)
        frame_id += 1

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]

            thumb = lm[4]
            index = lm[8]
            middle = lm[12]

            d_ti = norm_dist(thumb, index)
            d_tm = norm_dist(thumb, middle)
            d_im = norm_dist(index, middle)

            sx, sy = map_to_screen(index.x, index.y)
            sx, sy = ema.update((sx, sy))

            if abs(sx - prev_x) > 2 or abs(sy - prev_y) > 2:
                pyautogui.moveTo(int(sx), int(sy), _pause=False)
                prev_x, prev_y = sx, sy

            pinch_index = d_ti < PINCH_THRESH
            pinch_middle = d_tm < PINCH_THRESH
            scroll_mode = d_im < SCROLL_PAIR_THRESH and not pinch_index and not pinch_middle

            # Left Click / Drag
            if pinch_index and not state.pinching_index:
                state.pinching_index = True
                state.start_pinch()
            elif not pinch_index and state.pinching_index:
                if state.dragging:
                    pyautogui.mouseUp()
                    state.dragging = False
                elif state.pinch_duration() <= CLICK_TAP_MAX_SECONDS:
                    pyautogui.click()
                state.pinching_index = False
                state.pinch_start_time = None

            if state.pinching_index and not state.dragging and state.pinch_duration() >= DRAG_HOLD_SECONDS:
                pyautogui.mouseDown()
                state.dragging = True

            # Right Click
            if pinch_middle and not state.pinching_middle:
                state.pinching_middle = True
                state.start_pinch()
            elif not pinch_middle and state.pinching_middle:
                if state.pinch_duration() <= CLICK_TAP_MAX_SECONDS:
                    pyautogui.click(button="right")
                state.pinching_middle = False
                state.pinch_start_time = None

            # Scroll
            if scroll_mode:
                state.scroll_buffer.append(index.y)
                if len(state.scroll_buffer) >= 2:
                    dy = state.scroll_buffer[-2] - state.scroll_buffer[-1]
                    pyautogui.scroll(int(dy * 120))
            else:
                state.scroll_buffer.clear()

            # -------- Manual HUD Drawing (Python 3.12 Safe) --------
            if SHOW_HUD:
                for p in lm:
                    cx, cy = int(p.x * w), int(p.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        cv2.imshow("Advanced AI Virtual Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
