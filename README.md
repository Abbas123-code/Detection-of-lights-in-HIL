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

THRESHOLDS = {
    "left_headlight": 100, "right_headlight": 120,
    "left_indicator": 250, "right_indicator": 200,
    "left_day_light": 100, "right_day_light": 100,
    "left_brake": 120, "right_brake": 100,
    "back_light1": 90, "back_light2": 90,
}

ROIS = {
    "left_headlight":   (366, 310, 275, 120),
    "right_headlight":  (697, 302, 325, 144),
    "left_indicator":   (115, 432, 64, 32),
    "right_indicator":  (1159, 477, 68, 42),
    "left_day_light":   (233, 454, 243, 58),
    "right_day_light":  (858, 477, 256, 53),
    "left_brake":       (105, 365, 115, 114),
    "right_brake":      (1126, 399, 126, 135),
    "back_light1":      (174, 236, 165, 106),
    "back_light2":       (1020, 222, 180, 101),
}

SAVE_DIR = r"C:\Users\Aurelion\Desktop\cameratesting\Recordings"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 2. INITIALIZATION
# ============================================================

telemetry_per_second = []
last_logged_sec = -1
start_time = time.time()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = os.path.join(SAVE_DIR, f"diagnostic_{ts}.avi")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path, fourcc, 20.0, (1280, 720))

print("--- SYSTEM ONLINE ---")
print(f"Recording to: {video_path}")
print("Press Q to stop recording")

# ============================================================
# 3. CORE PROCESSING LOOP
# ============================================================

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time
        current_sec = int(elapsed_time)

        display_frame = frame.copy()

        cv2.rectangle(display_frame, (0, 0), (320, 260), (20, 20, 20), -1)
        cv2.putText(display_frame, f"SYSTEM TIME: {current_sec}s",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        current_states = {}

        for i, (name, (x, y, w, h)) in enumerate(ROIS.items()):
            roi_zone = frame[y:y+h, x:x+w]
            brightness = np.mean(
                cv2.cvtColor(roi_zone, cv2.COLOR_BGR2GRAY)
            ) if roi_zone.size > 0 else 0

            is_on = brightness > THRESHOLDS[name]
            current_states[name] = "ON" if is_on else "OFF"

            color = (0, 255, 0) if is_on else (0, 0, 255)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)

            y_offset = 65 + (i * 22)
            cv2.putText(display_frame, f"{name[:12].upper()}:",
                        (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1)
            cv2.putText(display_frame, current_states[name],
                        (180, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        if current_sec > last_logged_sec:
            telemetry_per_second.append({
                "second": current_sec,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "data": current_states
            })
            last_logged_sec = current_sec

        out.write(display_frame)
        cv2.imshow("Automotive Diagnostic HUD", display_frame)

        # ✅ CLEAN EXIT WITH Q
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("Stopping recording (Q pressed)")
            break

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # ============================================================
    # 4. FINAL EXPORT (JSON & GRAPH)
    # ============================================================

    json_path = os.path.join(SAVE_DIR, f"telemetry_{ts}.json")
    with open(json_path, 'w') as f:
        json.dump(telemetry_per_second, f, indent=4)

    # ===================== GRAPH =====================

    plt.figure(figsize=(16, 9), facecolor='#f7f7f7')

    spacing = 1.5
    yticks = []
    ylabels = []

    x_values = [entry["second"] for entry in telemetry_per_second]

    for idx, name in enumerate(ROIS):
        y_raw = [1 if entry["data"][name] == "ON" else 0
                 for entry in telemetry_per_second]
        y_offset = [val + idx * spacing for val in y_raw]

        plt.step(x_values, y_offset, where='post', linewidth=2)

        yticks.append(idx * spacing + 0.5)
        ylabels.append(name.replace("_", " ").upper())

    plt.title("Vehicle Light ON/OFF Timeline", fontsize=16, fontweight='bold')
    plt.xlabel("Time (Seconds)", fontsize=13)
    plt.ylabel("Vehicle Lights", fontsize=13)

    plt.xticks(x_values)
    plt.yticks(yticks, ylabels)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()

    graph_path = os.path.join(SAVE_DIR, f"analysis_{ts}.png")
    plt.savefig(graph_path)

    print("--- SESSION COMPLETE ---")
    print(f"JSON Report: {json_path}")
    print(f"Stability Graph: {graph_path}")
