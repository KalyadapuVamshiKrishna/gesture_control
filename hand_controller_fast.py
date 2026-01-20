import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import threading
import os
import json

# --- CONFIGURATION ---
DEFAULT_CONFIG = {
    'camera_id': 0,
    'smoothing': 5,
    'frame_reduction': 100,
    'click_dist': 30,
    'scroll_speed': 20,
    'hold_threshold': 0.2
}

class ThreadedGestureController:
    def __init__(self):
        # 1. Load Config
        if os.path.exists('gesture_config.json'):
            with open('gesture_config.json', 'r') as f: self.config = json.load(f)
        else: self.config = DEFAULT_CONFIG

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
        self.action_queue = None # "click", "right_click", "scroll_up", etc.
        self.running = True
        
        # State Flags
        self.drag_active = False
        self.drag_start = 0
        self.paused = False

        # Disable FailSafes for speed
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False

    def mouse_worker(self):
        """ This runs in a separate thread to handle Mouse Movement """
        while self.running:
            if not self.paused:
                # 1. Smoothing Logic (EMA)
                self.curr_x += (self.target_x - self.curr_x) / self.config['smoothing']
                self.curr_y += (self.target_y - self.curr_y) / self.config['smoothing']
                
                # 2. Move Mouse
                # We skip moving if the change is tiny (micro-jitter fix)
                if abs(self.target_x - self.curr_x) > 1 or abs(self.target_y - self.curr_y) > 1:
                    pyautogui.moveTo(self.curr_x, self.curr_y)

                # 3. Handle Actions (Clicks/Scrolls)
                if self.action_queue:
                    if self.action_queue == "click":
                        pyautogui.click()
                    elif self.action_queue == "right_click":
                        pyautogui.rightClick()
                    elif self.action_queue == "scroll_up":
                        pyautogui.scroll(self.config['scroll_speed'])
                    elif self.action_queue == "scroll_down":
                        pyautogui.scroll(-self.config['scroll_speed'])
                    elif self.action_queue == "drag_start":
                        pyautogui.mouseDown()
                    elif self.action_queue == "drag_end":
                        pyautogui.mouseUp()
                    
                    self.action_queue = None # Clear action
                    
            time.sleep(0.001) # Yield to CPU (Prevent overheating)

    def get_distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def start(self):
        # Start the Mouse Thread
        t = threading.Thread(target=self.mouse_worker)
        t.daemon = True
        t.start()

        print(">>> THREADED CONTROLLER STARTED <<<")
        print("Use 'q' to Quit. Spacebar to Pause.")

        while True:
            success, frame = self.cap.read()
            if not success: break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw Box
            cv2.rectangle(frame, (self.config['frame_reduction'], self.config['frame_reduction']), 
                          (self.w_cam - self.config['frame_reduction'], self.h_cam - self.config['frame_reduction']),
                          (255, 0, 255), 1)

            if results.multi_hand_landmarks and not self.paused:
                lm = results.multi_hand_landmarks[0].landmark
                
                # Get Finger Tips
                idx_x, idx_y = int(lm[8].x * self.w_cam), int(lm[8].y * self.h_cam)
                thumb_x, thumb_y = int(lm[4].x * self.w_cam), int(lm[4].y * self.h_cam)
                mid_y = int(lm[12].y * self.h_cam)

                # Detect Fingers Up (Simplified Y-Check)
                index_up = lm[8].y < lm[6].y
                middle_up = lm[12].y < lm[10].y
                pinky_up = lm[20].y < lm[18].y
                
                # --- LOGIC ---

                # 1. PAUSE (Fist)
                if not index_up and not middle_up and not pinky_up:
                    cv2.putText(frame, "PAUSED", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                    if self.drag_active: 
                        self.action_queue = "drag_end"
                        self.drag_active = False

                # 2. SCROLL (Index + Middle Up)
                elif index_up and middle_up:
                    cv2.putText(frame, "SCROLL", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
                    if idx_y < (self.h_cam/2) - 30: self.action_queue = "scroll_up"
                    elif idx_y > (self.h_cam/2) + 30: self.action_queue = "scroll_down"

                # 3. CURSOR (Index Up)
                elif index_up:
                    # Map Coordinates (Update Shared Variables for Thread)
                    target_x = np.interp(idx_x, (self.config['frame_reduction'], self.w_cam - self.config['frame_reduction']), (0, self.w_scr))
                    target_y = np.interp(idx_y, (self.config['frame_reduction'], self.h_cam - self.config['frame_reduction']), (0, self.h_scr))
                    
                    self.target_x = target_x
                    self.target_y = target_y

                    # Check Click
                    dist = self.get_distance((thumb_x, thumb_y), (idx_x, idx_y))
                    
                    if dist < self.config['click_dist']:
                        if self.drag_start == 0: self.drag_start = time.time()
                        
                        if time.time() - self.drag_start > self.config['hold_threshold']:
                            if not self.drag_active:
                                self.action_queue = "drag_start"
                                self.drag_active = True
                                cv2.circle(frame, (idx_x, idx_y), 10, (0, 255, 0), -1)
                    else:
                        if self.drag_active:
                            self.action_queue = "drag_end"
                            self.drag_active = False
                        elif self.drag_start > 0:
                            self.action_queue = "click"
                        self.drag_start = 0

            # Display
            cv2.imshow("Threaded Controller", frame)
            key = cv2.waitKey(1)
            if key == ord('q'): 
                self.running = False
                break
            elif key == 32: # Spacebar
                self.paused = not self.paused

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ThreadedGestureController()
    app.start()