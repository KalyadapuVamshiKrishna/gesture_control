import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import threading
import os
import json
import keyboard

# --- CONFIGURATION ---
DEFAULT_CONFIG = {
    'camera_id': 0,
    'smoothing_normal': 5,
    'smoothing_precision': 20,
    'click_dist': 30,
    'scroll_speed': 20,
    'frame_reduction': 100,
    'hold_time_threshold': 0.2, # Seconds to hold before dragging
    'swipe_threshold': 50, # Pixel movement to trigger swipe
    'auto_pause_delay': 60, # Seconds before auto-sleep
}

class ThreadedGestureController:
    def __init__(self):
        # 1. Load Config
        self.load_config()

        # 2. Setup MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 3. Setup Variables
        self.cap = cv2.VideoCapture(self.config['camera_id'])
        self.w_cam, self.h_cam = 640, 480
        self.cap.set(3, self.w_cam)
        self.cap.set(4, self.h_cam)
        self.w_scr, self.h_scr = pyautogui.size()
        
        # Shared Variables (Between Threads)
        self.target_x, self.target_y = 0, 0
        self.curr_x, self.curr_y = 0, 0
        self.action_queue = None # Tuple: ("action_name", data) or str "action_name"
        self.running = True
        self.paused = False
        
        # State Flags
        self.drag_active = False
        self.drag_start_time = 0
        self.last_action_time = 0 # For rate limiting
        self.prev_wrist_x = 0 # For swipe
        self.curr_smoothing = self.config['smoothing_normal']

        # PyAutoGUI Setup
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False

    def load_config(self):
        if os.path.exists('gesture_config.json'):
            try:
                with open('gesture_config.json', 'r') as f: self.config = json.load(f)
                # Merge with default to ensure all keys exist
                for key, val in DEFAULT_CONFIG.items():
                    if key not in self.config: self.config[key] = val
            except:
                self.config = DEFAULT_CONFIG
        else: 
            self.config = DEFAULT_CONFIG
            with open('gesture_config.json', 'w') as f: json.dump(self.config, f, indent=4)

    def get_distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def detect_fingers(self, lm):
        # 1=Up, 0=Down
        fingers = []
        
        # Thumb: Compare tip x to knuckle x (Tip ID 4 vs Knuckle ID 3)
        # Note: Assuming right hand behavior for simplicity or mirrored left. 
        # Ideally we use handedness but this simple check works for most selfie-mode interactions.
        if lm[4].x < lm[3].x: fingers.append(1) 
        else: fingers.append(0)

        # Other 4 Fingers (Tip y < Pip y)
        for id in [8, 12, 16, 20]:
            if lm[id].y < lm[id - 2].y: fingers.append(1)
            else: fingers.append(0)
        return fingers

    def mouse_worker(self):
        """ This runs in a separate thread to handle Mouse Movement and Input """
        while self.running:
            if not self.paused:
                # 1. Smoothing Logic (EMA)
                # Apply current smoothing factor
                self.curr_x += (self.target_x - self.curr_x) / self.curr_smoothing
                self.curr_y += (self.target_y - self.curr_y) / self.curr_smoothing
                
                # 2. Move Mouse
                # Micro-jitter fix: only move if difference is significant enough
                if abs(self.target_x - self.curr_x) > 1 or abs(self.target_y - self.curr_y) > 1:
                    try:
                        pyautogui.moveTo(self.curr_x, self.curr_y)
                    except pyautogui.FailSafeException:
                        pass # Ignore failsafe

                # 3. Handle Actions
                if self.action_queue:
                    action = self.action_queue
                    self.action_queue = None # Clear immediately to prevent double firing
                    
                    if action == "click": pyautogui.click()
                    elif action == "right_click": pyautogui.rightClick()
                    elif action == "drag_start": pyautogui.mouseDown()
                    elif action == "drag_end": pyautogui.mouseUp()
                    elif action == "scroll_up": pyautogui.scroll(self.config['scroll_speed'])
                    elif action == "scroll_down": pyautogui.scroll(-self.config['scroll_speed'])
                    elif action == "media_play_pause": pyautogui.press('playpause')
                    elif action == "volume_up": pyautogui.press('volumeup')
                    elif action == "volume_down": pyautogui.press('volumedown')
                    elif action == "swipe_left": pyautogui.press('left')
                    elif action == "swipe_right": pyautogui.press('right')
                    
            time.sleep(0.005) # High refresh rate for mouse thread

    def start(self):
        # Start Mouse Thread
        t = threading.Thread(target=self.mouse_worker)
        t.daemon = True
        t.start()

        print(">>> OPTIMIZED GESTURE CONTROLLER STARTED <<<")
        print("Spacebar: Toggle Pause | 'q': Quit")

        last_active_time = time.time()

        while True:
            # Check manual pause
            if keyboard.is_pressed('space'):
                self.paused = not self.paused
                time.sleep(0.3)
                
            # Auto-Pause
            if time.time() - last_active_time > self.config['auto_pause_delay']:
                if not self.paused:
                    print("Auto-pausing...")
                    self.paused = True

            success, frame = self.cap.read()
            if not success: break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw ROI
            cv2.rectangle(frame, (self.config['frame_reduction'], self.config['frame_reduction']), 
                          (self.w_cam - self.config['frame_reduction'], self.h_cam - self.config['frame_reduction']),
                          (255, 0, 255), 1)

            # UI Text
            status_text = "ACTIVE" if not self.paused else "PAUSED"
            color = (0, 255, 0) if not self.paused else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            if results.multi_hand_landmarks and not self.paused:
                last_active_time = time.time()
                
                # Only support one hand for now
                lm_obj = results.multi_hand_landmarks[0]
                lm = lm_obj.landmark
                
                fingers = self.detect_fingers(lm)
                
                # Key Coordinates
                idx_x, idx_y = int(lm[8].x * self.w_cam), int(lm[8].y * self.h_cam)
                thumb_x, thumb_y = int(lm[4].x * self.w_cam), int(lm[4].y * self.h_cam)
                pinky_x, pinky_y = int(lm[20].x * self.w_cam), int(lm[20].y * self.h_cam)
                wrist_x = int(lm[0].x * self.w_cam)

                # --- GESTURES ---
                
                # 1. KEYBOARD / SWIPE (All Fingers Up) 🖐️
                if fingers == [1, 1, 1, 1, 1]:
                    cv2.putText(frame, "SWIPE MODE", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
                    diff_x = wrist_x - self.prev_wrist_x
                    
                    if abs(diff_x) > self.config['swipe_threshold']:
                        if time.time() - self.last_action_time > 0.5: # Rate Stamp
                            if diff_x > 0: self.action_queue = "swipe_right"
                            else: self.action_queue = "swipe_left"
                            self.last_action_time = time.time()
                    
                    self.prev_wrist_x = wrist_x
                    if self.drag_active: # Release drag if hand opens
                        self.action_queue = "drag_end"
                        self.drag_active = False

                # 2. MEDIA CONTROL (Pinky + Ring) 🤘
                elif fingers == [0, 0, 0, 1, 1] or fingers == [1, 0, 0, 1, 1]:
                    cv2.putText(frame, "MEDIA", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 100, 100), 2)
                    if time.time() - self.last_action_time > 1.0:
                         self.action_queue = "media_play_pause"
                         self.last_action_time = time.time()

                # 3. RIGHT CLICK (Fist + Thumb) 👍
                elif fingers == [1, 0, 0, 0, 0]:
                    cv2.putText(frame, "RIGHT CLICK", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    if time.time() - self.last_action_time > 0.5:
                        self.action_queue = "right_click"
                        self.last_action_time = time.time()

                # 4. VOLUME (Pinky + Thumb) 🤙
                elif fingers[0] == 1 and fingers[4] == 1 and fingers[1:4] == [0, 0, 0]:
                    cv2.putText(frame, "VOLUME", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    dist = self.get_distance((thumb_x, thumb_y), (pinky_x, pinky_y))
                    if time.time() - self.last_action_time > 0.1: # Fast repeat
                        if dist > 150: self.action_queue = "volume_up"
                        elif dist < 60: self.action_queue = "volume_down"
                        self.last_action_time = time.time()

                # 5. SCROLL (Index + Middle) ✌️
                elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
                    cv2.putText(frame, "SCROLL", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                    if idx_y < (self.h_cam/2) - 40: self.action_queue = "scroll_up"
                    elif idx_y > (self.h_cam/2) + 40: self.action_queue = "scroll_down"

                # 6. CURSOR & CLICK (Index Up) ☝️
                elif fingers[1] == 1 and fingers[2] == 0:
                    # Precision Mode (Index + Pinky UP)
                    if fingers[4] == 1:
                        self.curr_smoothing = self.config['smoothing_precision']
                        cv2.putText(frame, "PRECISION", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    else:
                        self.curr_smoothing = self.config['smoothing_normal']

                    # Update Target Coordinates for Thread
                    raw_x = np.interp(idx_x, (self.config['frame_reduction'], self.w_cam - self.config['frame_reduction']), (0, self.w_scr))
                    raw_y = np.interp(idx_y, (self.config['frame_reduction'], self.h_cam - self.config['frame_reduction']), (0, self.h_scr))
                    self.target_x, self.target_y = raw_x, raw_y

                    # Click / Drag Logic
                    dist = self.get_distance((thumb_x, thumb_y), (idx_x, idx_y))
                    
                    if dist < self.config['click_dist']:
                        cv2.line(frame, (thumb_x, thumb_y), (idx_x, idx_y), (0, 255, 0), 2)
                        
                        if self.drag_start_time == 0: 
                            self.drag_start_time = time.time()
                        
                        # Check Hold Threshold for Drag
                        if time.time() - self.drag_start_time > self.config['hold_time_threshold']:
                            if not self.drag_active:
                                self.action_queue = "drag_start"
                                self.drag_active = True
                                cv2.circle(frame, (idx_x, idx_y), 15, (0, 255, 0), cv2.FILLED)
                    else:
                        # Released
                        if self.drag_active:
                            self.action_queue = "drag_end"
                            self.drag_active = False
                        elif self.drag_start_time > 0:
                            self.action_queue = "click"
                        
                        self.drag_start_time = 0

            # Display
            cv2.imshow("Optimized Gesture Controller", frame)
            key = cv2.waitKey(1)
            if key == ord('q'): break

        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ThreadedGestureController()
    app.start()
