import cv2
from deepface import DeepFace
import numpy as np
import os
from matplotlib import pyplot as plt

# Ensure these files are in the same directory as this script or provide the full paths
AGE_PROTO = 'deploy_age.prototxt'
AGE_MODEL = 'age_net.caffemodel'
GENDER_PROTO = 'deploy_gender.prototxt'
GENDER_MODEL = 'gender_net.caffemodel'

# Check if the files exist
for file in [AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]:
    if not os.path.isfile(file):
        raise FileNotFoundError(f"File not found: {file}")

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Function to detect and label mood, age, and gender
def detect_mood_age_gender(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Detect mood
        try:
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            mood = result.get('dominant_emotion', 'Unknown') if isinstance(result, dict) else "Unknown"
        except Exception as e:
            print(f"Error in DeepFace analysis: {e}")
            mood = "Error"

        # Create blob for age and gender prediction
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Detect age
        try:
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_BUCKETS[age_preds[0].argmax()]
        except Exception as e:
            print(f"Error in age prediction: {e}")
            age = "Unknown"

        # Detect gender
        try:
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]
        except Exception as e:
            print(f"Error in gender prediction: {e}")
            gender = "Unknown"

        # Draw rectangle and labels
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f'{mood}, Age: {age}, Gender: {gender}'
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

# Open camera and process frames
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_mood_age_gender(frame)
    
    # Convert BGR to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.show()

    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
        break

cap.release()
cv2.destroyAllWindows()
