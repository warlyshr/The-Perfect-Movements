# Eye Gesture Control (Lean, No-Speech Build)

A lightweight and real-time eye-gesture control system built using OpenCV, MediaPipe, and Python. This project allows users to control the mouse cursor and perform click actions with eye movements and blinks. It’s designed for responsiveness and accuracy with low-latency input.

## IMP:
If you need calibration and more accuracy and most convenience choose "Pupil-Tracker.py" but if you dont want calibration and let the system handle everything you have to choose "No-Calibration-Needed" file its less convenient but perfectly does the job 

###  Features

-  **Eye movement detection**: Control left and right movements of the cursor using eye gestures.
-  **Blink detection**: Click the mouse by blinking.
-  **Long blink detection**: Pause and resume command execution with a long blink.
-  **Fast and responsive**: Optimized for low-latency and real-time feedback.
-  **Debug window**: Provides real-time feedback on eye gestures and system status.

###  Getting Started

Follow these steps to run the project:

#### 1. Clone the repository:

```bash
git clone https://github.com/your-saad-rafeque/The-OG-Movements.git
cd The-OG-Movements
```
#### 2. Install dependencies:
Ensure you have Python 3.7+ installed, then install the required Python libraries:

```bash
pip install -r requirements.txt
```

#### 3. Run the script:
Make sure your camera is connected and positioned correctly for best results.
Run the Python script to start the eye gesture control:

```bash
python main.py
```


#### 4. Customize Settings:
You can adjust various settings in the script, including camera resolution (larger the size of resolution more processing power is required), blink thresholds, and movement speed (how many pixels you want to move across the screen after one cursor movements) by editing the USER SETTINGS section of the script.

### Notes
- Press Q to quit the application at any time or CTRL+C in case "Q" other technical problem arises.

- The script expects a working webcam and proper lighting for best performance.

- Calibration data is saved per session. Ensure proper calibration by following the on-screen instructions.

### Troubleshooting
-  "Cannot open camera": Make sure your webcam is connected, or update the camera index in the script (CAM_INDEX).

-  Low performance: Lower the frame rate (FPS_LIMIT) or resolution (CAM_WIDTH, CAM_HEIGHT) to improve performance.

### Customization
This project can be customized to support additional gestures, or even integrated into other software, such as media players or custom applications.
If you have reached here do me a favour by forking this repo ps or by giving it a star please

## Cheers :))
