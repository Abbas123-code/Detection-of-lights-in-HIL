# Detection-of-lights-in-HIL
import cv2
import numpy as np
import time
import os
import json
import csv
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# 1. CONFIGURATION & CALIBRATION
# ============================================================

# Intensity thresholds: If the mean pixel value in the ROI is above this, the light is "ON"
THRESHOLDS = {
    "left_headlight": 180, "right_headlight": 180,
    "left_indicator": 200, "right_indicator": 200,
    "left_day_light": 120, "right_day_light": 120,
    "left_brake": 180, "right_brake": 100,
}

# Regions of Interest (ROI): Defined as (x_start, y_start, width, height)
# These coordinates map specifically to where the lights appear in your camera frame
ROIS = {
    "left_headlight":   (366, 310, 275, 120),
    "right_headlight":  (697, 302, 325, 144),
    "left_indicator":   (115, 432, 64, 32),
    "right_indicator":  (1159, 477, 68, 42),
    "left_day_light":   (233, 454, 243, 58),
    "right_day_light":  (858, 477, 256, 53),
    "left_brake":       (105, 365, 115, 114),
    "right_brake":      (1126, 399, 126, 135),
}

# Set up the directory to store video recordings, JSON logs, and graph images
SAVE_DIR = r"C:\Users\Aurelion\Desktop\cameratesting\Recordings"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 2. INITIALIZATION
# ============================================================

# state_history: Stores the ON/OFF status for each light over time
state_history = {k: [] for k in ROIS}
# telemetry_per_second: List to store a snapshot of all lights every 1 second
telemetry_per_second = []
# Tracks the last second logged to prevent multiple logs within the same second
last_logged_sec = -1
# Record the exact start time of the system
start_time = time.time()

# Camera Input: Initialize the webcam/camera at 720p resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Video Setup: Generate unique filename using current date/time
ts = datetime.now().strftime("%Y%m%d")
video_path = os.path.join(SAVE_DIR, f"diagnostic_{ts}.avi")
# VideoWriter: Set codec (XVID) and frame rate (20 FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path, fourcc, 20.0, (1280, 720))

print(f"--- SYSTEM ONLINE ---")
print(f"Recording to: {video_path}")

# ============================================================
# 3. CORE PROCESSING LOOP
# ============================================================

try:
    while cap.isOpened():
        ret, frame = cap.read() # Capture individual frame from camera
        if not ret: break       # Exit if frame is not captured

        # Calculate how many seconds have passed since start
        elapsed_time = time.time() - start_time
        current_sec = int(elapsed_time)
        
        # Create a copy of the frame to draw the UI (prevents interference with raw data)
        display_frame = frame.copy()
        
        # HUD: Draw a dark semi-transparent box for the text status area
        cv2.rectangle(display_frame, (0, 0), (320, 260), (20, 20, 20), -1)
        # HUD: Show current system runtime
        cv2.putText(display_frame, f"SYSTEM TIME: {current_sec}s", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        current_states = {} # Dictionary to store light states for the current frame

        # Loop through each light defined in ROIS
        for i, (name, (x, y, w, h)) in enumerate(ROIS.items()):
            # Logic: Crop the current frame to the specific light area
            roi_zone = frame[y:y+h, x:x+w]
            # Logic: Convert that area to grayscale and calculate the average brightness
            brightness = np.mean(cv2.cvtColor(roi_zone, cv2.COLOR_BGR2GRAY)) if roi_zone.size > 0 else 0
            
            # Logic: Check if average brightness exceeds the defined threshold
            is_on = brightness > THRESHOLDS[name]
            current_states[name] = "ON" if is_on else "OFF"
            
            # Visuals: Determine color (Green for ON, Red for OFF)
            color = (0, 255, 0) if is_on else (0, 0, 255)
            # Visuals: Draw the rectangle around the light on the screen
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            
            # HUD: List the light name and its status (ON/OFF) in the status box
            y_offset = 65 + (i * 22)
            cv2.putText(display_frame, f"{name[:12].upper()}:", (15, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display_frame, current_states[name], (180, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Logic: If 1 second has passed, take a snapshot of all light states for the JSON report
        # This is the main logic it basically sends info at every one second to json format 
        # Instead of just recording when a light changes, this code uses a Heartbeat Logic.
        #  It creates a "snapshot" of all lights at every 1-second interval. 
        # This results in a stable JSON file where you can see exactly what every light was doing at second 1, second 2, second 3, etc.
        if current_sec > last_logged_sec:
            telemetry_per_second.append({
                "second": current_sec,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "data": current_states
            })
            last_logged_sec = current_sec # Update last logged second

        # Write the HUD-processed frame to the video file
        out.write(display_frame)
        # Show the live video window to the user
        cv2.imshow("Automotive Diagnostic HUD", display_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Safe Exit: Ensure camera and video file are closed correctly even if an error occurs
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # ============================================================
    # 4. FINAL EXPORT (JSON & GRAPH)
    # ============================================================
    
    # Save the per-second snapshots into a structured JSON file
    json_path = os.path.join(SAVE_DIR, f"telemetry_{ts}.json")
    with open(json_path, 'w') as f:
        json.dump(telemetry_per_second, f, indent=4)

    # Visualization: Plot the light states over time
    plt.figure(figsize=(14, 7), facecolor='#f0f0f0')
    for name in ROIS:
        # Convert "ON/OFF" strings into 1 and 0 for graphing
        y_values = [1 if entry["data"][name] == "ON" else 0 for entry in telemetry_per_second]
        x_values = [entry["second"] for entry in telemetry_per_second]
        # Draw a step-style plot (ideal for digital ON/OFF signals)
        plt.step(x_values, y_values, label=name, where='post', linewidth=1.5)

    # Graph Formatting
    plt.title("Vehicle Light Stability & Signal Timeline", fontsize=14, fontweight='bold')
    plt.xlabel("Time (Seconds)", fontsize=12)
    plt.ylabel("Logic State (ON/OFF)", fontsize=12)
    plt.yticks([0, 1], ["OFF", "ON"])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the final graph as a PNG image
    graph_path = os.path.join(SAVE_DIR, f"analysis_{ts}.png")
    plt.savefig(graph_path)
    
    print(f"--- SESSION COMPLETE ---")
    print(f"JSON Report: {json_path}")
    print(f"Stability Graph: {graph_path}")
