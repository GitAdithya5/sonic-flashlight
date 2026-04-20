import cv2
import time
import subprocess
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo26n.pt")

print("=== Sonic Flashlight - Stage 1 ===")
print("Press SPACEBAR to scan. Press Q to quit.")

# ✅ ESP32 OV3660 stream — replace MacBook webcam
ESP32_STREAM_URL = "http://192.168.6.180/stream"  # try /cam-hi.jpg or /?action=stream if this fails
cap = cv2.VideoCapture(ESP32_STREAM_URL)

if not cap.isOpened():
    print("ERROR: Could not open ESP32 stream. Check IP and that camera is running.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("WARNING: Frame drop from ESP32 stream, retrying...")
        cap.open(ESP32_STREAM_URL)  # auto-reconnect
        continue

    display = frame.copy()
    cv2.putText(display, "SONIC FLASHLIGHT - Press SPACE to scan",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Sonic Flashlight - Stage 1", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        print("\n[SCAN TRIGGERED]")
        start_time = time.time()

        results = model(frame, verbose=False, conf=0.15)
        names = model.names
        detected = []
        for r in results:
            for cls in r.boxes.cls:
                label = names[int(cls)]
                if label not in detected:
                    detected.append(label)

        end_time = time.time()
        latency = round(end_time - start_time, 3)

        if detected:
            speech = ", ".join(detected)
            print(f"Detected: {speech}")
            print(f"Latency:  {latency}s")
            annotated = results[0].plot()
            cv2.imshow("Sonic Flashlight - Stage 1", annotated)
            cv2.waitKey(1500)
            subprocess.run(["say", speech])
        else:
            print("Nothing detected.")
            print(f"Latency:  {latency}s")
            subprocess.run(["say", "Nothing detected"])

    elif key == ord('q'):
        print("\n[EXITING]")
        break

cap.release()
cv2.destroyAllWindows()