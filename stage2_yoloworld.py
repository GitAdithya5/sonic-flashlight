import cv2
import time
import subprocess
from ultralytics import YOLO

# Load YOLO-World model — larger but detects far more classes
model = YOLO("yolov8x-worldv2.pt")

# Define all the classes you want to detect
# Add or remove anything from this list
model.set_classes([
    # COCO basics
    "person", "chair", "couch", "bed", "dining table", "toilet",
    "laptop", "phone", "keyboard", "mouse", "remote", "tv",
    "bottle", "cup", "bowl", "knife", "fork", "spoon",
    "backpack", "handbag", "umbrella", "book", "clock",
    "scissors", "teddy bear", "vase", "toothbrush",
    "car", "bicycle", "motorcycle", "bus", "truck",
    "cat", "dog", "bird",

    # Extra classes COCO doesn't have
    "door", "door handle", "door knob",
    "stairs", "step",
    "light switch", "power socket",
    "fan", "ceiling fan",
    "window", "curtain", "pillar",
    "shoe", "slipper", "sandal",
    "plate", "tiffin box", "glass",
    "pen", "pencil", "ruler",
    "wallet", "keys",
    "indian rupee note", "coin",
    "helmet", "mask",
    "pillow", "blanket",
    "dustbin", "bucket",
    "rope", "wire",
    "fire extinguisher",
    "wheelchair", "walking stick", "cane",
])

print("=== Sonic Flashlight - YOLO-World (600+ classes) ===")
print("Press SPACEBAR to scan. Press Q to quit.")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    cv2.putText(display, "YOLO-WORLD MODE - Press SPACE to scan",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.imshow("Sonic Flashlight - YOLO World", display)

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
