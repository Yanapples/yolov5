import cv2
from ultralytics import YOLO
import face_recognition
import numpy as np
import torch
import pickle
import time

# Load known face encodings
with open("yolov8_db.pickle", "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)
print(f"Loaded {len(known_face_names)} faces from database.")

# Set device for YOLO
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO('yolov5n.pt')
model.to(device)

# Initialize video
video_path = "input_video.mp4"
video = cv2.VideoCapture(video_path)

frame_count = 0
RECOGNITION_INTERVAL = 1  # Only do face recognition every N frames

# Track recognized face names across frames to reduce flickering
last_recognized_names = {}
face_locations = []
face_names = []

t0 = time.time()
n_frames = 1

while True:
    ret, frame = video.read()
    if not ret:
        break

    results = model(frame)
    frame_count += 1
    do_face_recognition = frame_count % RECOGNITION_INTERVAL == 0

    for result in results:
        for box in result.boxes:
            # Convert box to int coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            name = None

            if do_face_recognition:
                # Crop person region from the frame
                person_frame = frame[y1:y2, x1:x2]

                # Convert cropped person region to RGB
                rgb_person = np.ascontiguousarray(person_frame[:, :, ::-1])

                # Detect faces in the person crop
                face_locations = face_recognition.face_locations(rgb_person, model='hog')
                face_encodings = face_recognition.face_encodings(rgb_person, face_locations, num_jitters=1)

                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_person, face_locations)
                    face_names = []
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Unknown"

                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]

                        face_names.append(name)

                    # Check if the name has changed or if it's the same person as last time
                    if name in last_recognized_names and last_recognized_names[name] >= 2:  # Consistent recognition count
                        name = last_recognized_names[name]  # Keep previous name if consistent
                    else:
                        last_recognized_names[name] = 0  # Reset recognition count if not consistent
                    # Increment name's recognition count
                    last_recognized_names[name] += 1

            # Draw bounding boxes and names on original full frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Adjust face box to match full frame coordinates
                top += y1
                bottom += y1
                left += x1
                right += x1

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    elapsed_time = time.time() - t0
    average_fps = n_frames / elapsed_time
    n_frames += 1
    cv2.rectangle(frame, (0,0), (550,50), (0,0,0), -1)
    cv2.rectangle(frame, (0,50), (200,100), (0,0,0), -1)
    cv2.putText(frame,f"YOLOv5n + HOG" , (0,40) , cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 2, cv2.LINE_AA)
    cv2.putText(frame,f"FPS :{'%s' % float('%.4g' % average_fps)}" , (0,90) , cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 2, cv2.LINE_AA)
        
    cv2.imshow('YOLOv5 + HOG', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
