import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import json
import keyboard 
import os

# --- DEFAULT CONFIGURATION (Will be saved to json) ---
DEFAULT_CONFIG = {
    'camera_id': 0,
    'smoothing_normal': 5,
    'smoothing_precision': 20,
    'click_dist': 30,
    'scroll_speed': 20,
    'frame_reduction': 100,
    'hold_time_threshold': 0.2, # Seconds to hold before dragging
    'double_click_speed': 0.3,
    'auto_pause_delay': 30, # Seconds before auto-sleep
    'swipe_threshold': 50 # Pixel movement to trigger swipe
}

# --- CLASS STRUCTURE FOR STATE MANAGEMENT ---
class GestureController:
    def __init__(self):
        self.load_config()
        
        # PyAutoGUI Setup
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False
        
        # MediaPipe Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0, # Fastest
            min_detection_confidence=0.5, # Lowered for reliability
            min_tracking_confidence=0.5
        )
        
        # State Variables
        self.enabled = True
        self.last_active_time = time.time()
        self.drag_start_time = 0
        self.drag_active = False
        self.prev_x, self.prev_y = 0, 0 # For smoothing
        self.prev_wrist_x = 0 # For swipe detection
        self.curr_smoothing = self.config['smoothing_normal']
        
        # Screen Info
        self.w_scr, self.h_scr = pyautogui.size()

    def setup_debug_window(self):
        cv2.namedWindow("Tuning Panel")
        cv2.resizeWindow("Tuning Panel", 400, 300)
        
        def nothing(x): pass

        # Create Sliders (Trackbars)
        # Name, Window, Default Value, Max Value, Callback
        cv2.createTrackbar("Click Dist", "Tuning Panel", self.config['click_dist'], 100, nothing)
        cv2.createTrackbar("Smoothing", "Tuning Panel", self.config['smoothing_normal'], 50, nothing)
        cv2.createTrackbar("ROI Size", "Tuning Panel", self.config['frame_reduction'], 200, nothing)
        cv2.createTrackbar("Scroll Speed", "Tuning Panel", self.config['scroll_speed'], 100, nothing)
        
    def load_config(self):
        if os.path.exists('gesture_config.json'):
            with open('gesture_config.json', 'r') as f:
                self.config = json.load(f)
        else:
            self.config = DEFAULT_CONFIG
            with open('gesture_config.json', 'w') as f:
                json.dump(self.config, f, indent=4)
        print("Config Loaded:", self.config)

    def get_distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def detect_fingers(self, lm, hand_label):
        # 1=Up, 0=Down
        fingers = []
        
        # Thumb Logic (Adaptive based on hand label left/right)
        # Note: MediaPipe mirrors input. "Right" hand usually appears as "Left" in selfie mode.
        # We use a simplified x-check relative to knuckle (Tip ID 4 vs Knuckle ID 3)
        if lm[4].x < lm[3].x: # Typical for Right Hand
            fingers.append(1) 
        else: 
            fingers.append(0)

        # 4 Fingers (Tip ID vs Pip ID)
        for id in [8, 12, 16, 20]:
            if lm[id].y < lm[id - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def run(self):
        cap = cv2.VideoCapture(self.config['camera_id'])
        w_cam, h_cam = 640, 480
        cap.set(3, w_cam)
        cap.set(4, h_cam)
        
        # ACTIVATE DEBUG TOOLS
        self.setup_debug_window()
        
        print(">>> PRO GESTURE CONTROLLER ACTIVE <<<")
        print("Spacebar: Toggle On/Off | 'q': Quit")

        while True:
            # 1. Toggle Check
            if keyboard.is_pressed('space'):
                self.enabled = not self.enabled
                time.sleep(0.3) # Debounce
                
            # 2. Auto-Pause Check
            if time.time() - self.last_active_time > self.config['auto_pause_delay']:
                if self.enabled:
                    print("Auto-Pausing due to inactivity...")
                    self.enabled = False

            # 3. Read Debug Sliders (Real-time Tuning)
            self.config['click_dist'] = cv2.getTrackbarPos("Click Dist", "Tuning Panel")
            s_val = cv2.getTrackbarPos("Smoothing", "Tuning Panel")
            self.config['smoothing_normal'] = s_val if s_val > 0 else 1
            self.config['frame_reduction'] = cv2.getTrackbarPos("ROI Size", "Tuning Panel")
            self.config['scroll_speed'] = cv2.getTrackbarPos("Scroll Speed", "Tuning Panel")

            success, frame = cap.read()
            if not success: break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw ROI
            cv2.rectangle(frame, (self.config['frame_reduction'], self.config['frame_reduction']), 
                          (w_cam - self.config['frame_reduction'], h_cam - self.config['frame_reduction']),
                          (255, 0, 255), 2)
            
            # Status Overlay
            status_text = "ACTIVE" if self.enabled else "DISABLED (Space to toggle)"
            color = (0, 255, 0) if self.enabled else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            if results.multi_hand_landmarks and self.enabled:
                self.last_active_time = time.time() # Reset inactivity timer
                
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    lm = hand_landmarks.landmark
                    fingers = self.detect_fingers(lm, handedness.classification[0].label)
                    
                    # Coordinates
                    idx_x, idx_y = int(lm[8].x * w_cam), int(lm[8].y * h_cam)
                    thumb_x, thumb_y = int(lm[4].x * w_cam), int(lm[4].y * h_cam)
                    pinky_x, pinky_y = int(lm[20].x * w_cam), int(lm[20].y * h_cam)
                    wrist_x = int(lm[0].x * w_cam)

                    # --- GESTURE LOGIC ---

                    # 1. KEYBOARD / SWIPE MODE (Open Hand: 5 Fingers Up) 🖐️
                    if fingers == [1, 1, 1, 1, 1]:
                        cv2.putText(frame, "KEYBOARD MODE", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
                        
                        # Detect Swipe Velocity
                        diff_x = wrist_x - self.prev_wrist_x
                        if abs(diff_x) > self.config['swipe_threshold']:
                            if diff_x > 0: pyautogui.press('right')
                            else: pyautogui.press('left')
                            time.sleep(0.3) # Debounce swipe
                        
                        self.prev_wrist_x = wrist_x # Update for next frame
                        if self.drag_active: pyautogui.mouseUp(); self.drag_active = False

                    # 2. MEDIA CONTROL (Pinky + Ring Up) 🤘 (Modified)
                    elif fingers == [0, 0, 0, 1, 1] or fingers == [1, 0, 0, 1, 1]:
                        cv2.putText(frame, "MEDIA", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 100, 100), 2)
                        if time.time() % 2.0 < 0.1: # rudimentary limit
                             pyautogui.press('playpause')
                             time.sleep(0.5)

                    # 3. RIGHT CLICK (Fist + Thumb Up) 👍
                    elif fingers == [1, 0, 0, 0, 0]:
                        cv2.putText(frame, "RIGHT CLICK", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                        pyautogui.rightClick()
                        time.sleep(0.5)

                    # 4. VOLUME (Pinky + Thumb Up) 🤙
                    elif fingers[0] == 1 and fingers[4] == 1 and fingers[1:4] == [0, 0, 0]:
                        cv2.putText(frame, "VOLUME", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                        dist = self.get_distance((thumb_x, thumb_y), (pinky_x, pinky_y))
                        
                        if dist > 150:
                            pyautogui.press('volumeup')
                        elif dist < 60:
                            pyautogui.press('volumedown')

                    # 5. SCROLL (Index + Middle Up) ✌️
                    elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
                        cv2.putText(frame, "SCROLL", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                        if idx_y < (h_cam/2) - 40: pyautogui.scroll(self.config['scroll_speed'])
                        elif idx_y > (h_cam/2) + 40: pyautogui.scroll(-self.config['scroll_speed'])

                    # 6. CURSOR & CLICK (Index Up) ☝️
                    elif fingers[1] == 1 and fingers[2] == 0:
                        # --- MOVEMENT ---
                        # Precision Check (Index + Pinky = Precision)
                        if fingers[4] == 1: 
                            self.curr_smoothing = self.config['smoothing_precision']
                            cv2.putText(frame, "PRECISION", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                        else: 
                            self.curr_smoothing = self.config['smoothing_normal']

                        # Coordinate Mapping
                        target_x = np.interp(idx_x, (self.config['frame_reduction'], w_cam - self.config['frame_reduction']), (0, self.w_scr))
                        target_y = np.interp(idx_y, (self.config['frame_reduction'], h_cam - self.config['frame_reduction']), (0, self.h_scr))
                        
                        # Smoothing (Exponential Moving Average)
                        curr_x = self.prev_x + (target_x - self.prev_x) / self.curr_smoothing
                        curr_y = self.prev_y + (target_y - self.prev_y) / self.curr_smoothing
                        
                        pyautogui.moveTo(curr_x, curr_y)
                        self.prev_x, self.prev_y = curr_x, curr_y

                        # --- CLICK / DRAG LOGIC WITH VISUAL HUD ---
                        dist = self.get_distance((thumb_x, thumb_y), (idx_x, idx_y))
                        
                        # Draw Debug Lines
                        if dist < self.config['click_dist']:
                            line_color = (0, 255, 0) # Green (Clicking)
                            hud_text = "CLICK"
                        else:
                            line_color = (0, 0, 255) # Red (Hovering)
                            hud_text = f"Dist: {int(dist)}"
                            
                        cv2.line(frame, (thumb_x, thumb_y), (idx_x, idx_y), line_color, 2)
                        cv2.putText(frame, hud_text, (thumb_x + 20, thumb_y), cv2.FONT_HERSHEY_PLAIN, 1, line_color, 1)

                        if dist < self.config['click_dist']:
                            if self.drag_start_time == 0:
                                self.drag_start_time = time.time() # Start Timer
                            
                            # Only trigger if held for X seconds
                            if time.time() - self.drag_start_time > self.config['hold_time_threshold']:
                                if not self.drag_active:
                                    pyautogui.mouseDown()
                                    self.drag_active = True
                                    cv2.circle(frame, (idx_x, idx_y), 15, (0, 255, 0), cv2.FILLED)
                        else:
                            # If released
                            if self.drag_active:
                                pyautogui.mouseUp()
                                self.drag_active = False
                            elif self.drag_start_time > 0:
                                # Quick tap detected (released before drag threshold)
                                pyautogui.click()
                            
                            self.drag_start_time = 0 # Reset Timer

            cv2.imshow("Pro Gesture Controller", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = GestureController()
    app.run()