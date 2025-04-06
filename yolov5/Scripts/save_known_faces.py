import cv2
from ultralytics import YOLO
import face_recognition
import numpy as np
import os
import pickle

known_faces_dir = 'images_processed'

known_face_encodings = []
known_face_names = []
            
for classname in os.listdir(known_faces_dir):
    for filename in os.listdir(f'{known_faces_dir}/{classname}'):
        image_path = f'{known_faces_dir}/{classname}/{filename}'
        image = face_recognition.load_image_file(image_path)
        
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(classname)
            
with open('known_face_encodings.pickle', 'wb') as file:
    pickle.dump(known_face_encodings, file)
    
with open('known_face_names.pickle', 'wb') as file:
    pickle.dump(known_face_names, file)