import cv2
from ultralytics import YOLO
import face_recognition
import numpy as np
import os
import pickle

model = YOLO('best_face_detection_yolov5n.pt')

known_face_encodings = []
known_face_names = []
            
with open('known_face_encodings.pickle', 'rb') as file:
    known_face_encodings = pickle.load(file)
            
with open('known_face_names.pickle', 'rb') as file:
    known_face_names = pickle.load(file)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_frame = frame[y1:y2, x1:x2]
                
                rgb_small_frame = np.ascontiguousarray(frame[:, :, ::-1])
                
                face_locations = face_recognition.face_locations(rgb_small_frame)
                
                if face_locations is not None and face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    face_names = []
                    for face_encoding in face_encodings:
                        
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Unknown"
                        
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                            print(name)
                            
                        face_names.append(name)
                        
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        
                        # confidence = box.conf[0]
                        # cv2.putText(frame, f'{name} {confidence:.2f}', (left, bottom - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)
                    
    cv2.imshow('YOLOv5n Object Detection with face-recognition Enhancement', frame)
    
    if cv2.waitKeyEx(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()