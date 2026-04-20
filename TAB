import cv2
import time
import subprocess
from ultralytics import YOLO

# Load YOLO26 Nano model (downloads automatically on first run)
model = YOLO("yolo26n.pt")

# Open MacBook webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("=== Sonic Flashlight - Stage 1 ===")
print("Press SPACEBAR to scan. Press Q to quit.")

last_spoken = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show live feed
    cv2.imshow("Sonic Flashlight - Stage 1", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        print("\n[SCAN TRIGGERED]")
        start_time = time.time()

        # Run inference
        results = model(frame, verbose=False)

        # Extract unique object names
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
            # Speak using macOS SAY
            subprocess.run(["say", speech])
        else:
            print("Nothing detected.")
            subprocess.run(["say", "Nothing detected"])

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

