import subprocess
import cv2
from ultralytics import YOLO
import time
from SpeedDetect.tracker import Tracker
import os

# Load YOLO model
model = YOLO(r'/yolov8s.pt')
model2 = YOLO(r'/models/best.pt')

# Open video file
cap = cv2.VideoCapture(r'/Sample_Videos/veh3.webm')

# Load class names
with open(r'/coco.txt', 'r') as my_file:
    class_list = my_file.read().split('\n')

# Initialize tracker object
tracker = Tracker()

ct = 0
cy1 = 322
cy2 = 368
offset = 6
v_d = {}
cnt = []
count = 0

output_dir = r'C:\Users\shara\OneDrive\Desktop\pythonProject\SpeedsterVision\output_bbox_images'
os.makedirs(output_dir, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 11 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model2.predict(frame)
    a = results[0].boxes.data
    coords = a[:]
    bbox = []

    for row in coords:
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        c = int(row[5])

        # Draw bounding box around car
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        bbox.append([x1, y1, x2, y2, c])  # Append the class id as well

    bbox2 = tracker.update(bbox)

    for item in bbox2:
        x3, y3, x4, y4, id = item
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        if cy1 - offset <= cy <= cy1 + offset:
            v_d[id] = time.time()

        if id in v_d:
            if cy2 - offset <= cy <= cy2 + offset:
                elps_time = time.time() - v_d[id]

                if id not in cnt:
                    cnt.append(id)
                    distance = 10
                    vech_speed = distance / elps_time
                    vech_speed = vech_speed * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, str(int(vech_speed)) + ' KMPH', (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                    if vech_speed > 10:
                        bbox_img = frame[y3:y4, x3:x4]
                        if bbox_img.size > 0:  # Ensure the bounding box is valid
                            bbox_filename = os.path.join(output_dir, f'{id}_{time.time()}.jpg')
                            cv2.imwrite(bbox_filename, bbox_img)
                        cv2.putText(frame, 'OVERSPEEDING', (25, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    # Draw checkpoint lines and text
    cv2.line(frame, (241, cy1), (823, cy1), (255, 255, 255), 1)
    cv2.putText(frame, 'CHKPT1', (281, 317), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(frame, 'CHKPT2', (187, 360), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 255), 1)
    cv2.line(frame, (142, cy2), (927, cy2), (255, 255, 255), 1)

    cv2.imshow("SpeedDetect", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
subprocess.run(['python' , 'OCR.py' , output_dir])
