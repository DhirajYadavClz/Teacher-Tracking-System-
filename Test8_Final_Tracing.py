import cv2
import time
import csv
from datetime import datetime
import numpy as np
import face_recognition
from ultralytics import YOLO

# -------------------------------
# CONFIG
# -------------------------------
GRACE_PERIOD = 3          # seconds to wait before pausing timer
FACE_CHECK_INTERVAL = 10  # frames to skip face recognition
CSV_FILE = "teaching_log.csv"

# -------------------------------
# LOAD MODELS
# -------------------------------
model = YOLO("yolov8n.pt")

# Load teacher face
teacher_image = face_recognition.load_image_file("teacher.jpg")
teacher_encoding = face_recognition.face_encodings(teacher_image)[0]

# -------------------------------
# CAMERA
# -------------------------------
cap = cv2.VideoCapture(0)

# -------------------------------
# STATE VARIABLES
# -------------------------------
teacher_track_id = None
teacher_present = False
start_time = None
total_time = 0.0
last_seen_time = None
lecture_start_time = None
lecture_end_time = None

frame_count = 0
prev_time = time.time()

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------------------------------
    # YOLO tracking
    # -------------------------------
    results = model.track(
        frame,
        persist=True,
        conf=0.5,
        verbose=False
    )

    current_ids = []

    # -------------------------------
    # Face recognition (optimized)
    # -------------------------------
    if frame_count % FACE_CHECK_INTERVAL == 0:
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    else:
        face_locations = []
        face_encodings = []

    if results[0].boxes is not None:
        for box in results[0].boxes:
            if int(box.cls[0]) != 0 or box.id is None:
                continue

            track_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_ids.append(track_id)

            # -------------------------------
            # Bind teacher ID if not set
            # -------------------------------
            if teacher_track_id is None:
                for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
                    cx = (left + right) // 2
                    cy = (top + bottom) // 2

                    if x1 < cx < x2 and y1 < cy < y2:
                        match = face_recognition.compare_faces(
                            [teacher_encoding],
                            face_enc,
                            tolerance=0.5
                        )
                        if match[0]:
                            teacher_track_id = track_id
                            lecture_start_time = datetime.now()  # record start of lecture

            # -------------------------------
            # Draw box
            # -------------------------------
            color = (255, 0, 0)
            label = f"ID: {track_id}"

            if track_id == teacher_track_id:
                color = (0, 255, 0)
                label = "TEACHER"
                last_seen_time = time.time()

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # -------------------------------
    # Time logic with grace period
    # -------------------------------
    now = time.time()

    if teacher_track_id in current_ids:
        if not teacher_present:
            teacher_present = True
            start_time = now
    else:
        if teacher_present and last_seen_time:
            if now - last_seen_time > GRACE_PERIOD:
                teacher_present = False
                total_time += now - start_time
                start_time = None
                teacher_track_id = None  # allow rebind
                lecture_end_time = datetime.now()  # record end of lecture

    # -------------------------------
    # FPS
    # -------------------------------
    fps = 1 / (now - prev_time)
    prev_time = now

    # -------------------------------
    # Display time
    # -------------------------------
    display_time = total_time
    if teacher_present and start_time is not None:
        display_time += now - start_time

    cv2.putText(frame, f"Teaching Time: {int(display_time)} sec",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Day 8 - Final System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# FINALIZE TIME BEFORE EXIT
# -------------------------------
if teacher_present and start_time is not None:
    total_time += time.time() - start_time
    lecture_end_time = datetime.now()

# -------------------------------
# SAVE CSV LOG
# -------------------------------
with open(CSV_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        datetime.now().strftime("%Y-%m-%d"),
        int(total_time),
        lecture_start_time.strftime("%H:%M:%S") if lecture_start_time else "",
        lecture_end_time.strftime("%H:%M:%S") if lecture_end_time else ""
    ])

cap.release()
cv2.destroyAllWindows()

print("Teaching time saved to", CSV_FILE)
print(f"Total Teaching Time: {int(total_time)} sec")
print(f"Lecture Start: {lecture_start_time}")
print(f"Lecture End: {lecture_end_time}")
