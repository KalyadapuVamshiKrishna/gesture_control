🖐️ AI Gesture Controller (Threaded & Zero-Lag)

A high-performance, computer vision-based mouse controller that uses hand gestures to interact with your OS. Unlike basic tutorials, this project uses a Multi-Threaded Architecture to separate camera processing from mouse actions, ensuring zero latency even on lower-end hardware.

It features a unique "Sniper Freeze" logic that locks the cursor when you attempt to click, solving the common "jitter" issue found in most touch-free interfaces.

🚀 Key Features

Zero-Lag Architecture: Uses threading to decouple the computer vision loop (30-60 FPS) from the mouse control loop.

Sniper Freeze: Automatically detects when you are "aiming" a click and freezes the cursor for pixel-perfect precision.

Dynamic Smoothing: Uses Exponential Moving Average (EMA) to filter out hand tremors without making the mouse feel "floaty."

Full OS Control: Supports Clicking, Drag & Drop, Scrolling, Right-Click, Volume Control, and Media Playback.

Customizable: Auto-generates a gesture_config.json file to tune speeds, sensitivity, and thresholds.

🛠️ Installation

Prerequisites

Python 3.7 or higher

A Webcam

1. Clone the Repository

git clone [https://github.com/yourusername/gesture-controller.git](https://github.com/yourusername/gesture-controller.git)
cd gesture-controller


2. Install Dependencies

pip install opencv-python mediapipe pyautogui numpy keyboard


⚡ Usage

Run the main script:

python hand_controller_threaded.py


Note: On Windows/Linux, you may need to run your terminal as Administrator/Sudo because the keyboard library hooks into global key events.

Controls

Spacebar: Toggle the system PAUSE/RESUME instantly.

'q': Quit the application.

✌️ Gesture Guide

Gesture

Action

Visual Cue

Index Finger Up ☝️

Move Cursor

Standard movement.

Pinch (Quick) 👌

Left Click

Tap Thumb & Index together.

Pinch (Hold) ✊

Drag & Drop

Hold pinch for 0.2s. Cursor turns Green.

Index + Middle Up ✌️

Scroll

Move hand Up/Down to scroll.

Index + Pinky Up 🤘

Precision Mode

Slower mouse speed for detailed work.

Fist + Thumb 👍

Right Click

Make a fist, thumb out.

Thumb + Pinky 🤙

Volume Control

Wide hand = Vol Up, Closed hand = Vol Down.

Pinky + Ring 🤟

Play/Pause

Toggles media playback.

Open Hand 🖐️

Swipe / Arrows

Flick wrist Left/Right for Arrow Keys.

Fist (No Thumb) ✊

Pause

Stops tracking (Relax mode).

⚙️ Configuration

On the first run, the app generates gesture_config.json. You can edit this file to tune the experience:

{
    "camera_id": 0,
    "smoothing_normal": 5,        // Higher = Smoother, Lower = Faster
    "click_dist": 30,             // Pixel distance to trigger a click
    "scroll_speed": 20,           // How fast the page scrolls
    "frame_reduction": 100,       // Padding around the camera frame
    "hold_time_threshold": 0.2,   // Seconds to hold before "Drag" activates
    "auto_pause_delay": 60        // Seconds of inactivity before sleep
}


🧠 How "Sniper Freeze" Works

One of the hardest parts of gesture control is clicking without moving the mouse (because your hand shakes when you pinch).

This project solves it with a Freeze Zone:

Move Zone (> 45px): Standard tracking.

Freeze Zone (30px - 45px): When your thumb gets close to your index finger, the code locks the cursor position. You can wiggle your finger, but the cursor stays still.

Click Zone (< 30px): The click fires exactly where you locked it.

🤝 Contributing

Feel free to open issues or submit pull requests. Ideas for future upgrades:

Custom gesture recording.

Virtual keyboard overlay.

Mac/Linux specific optimizations.

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.