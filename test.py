import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from win32com.client import Dispatch

def speak(message):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(message)

# Load pre-trained face detection model
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load face data and labels
try:
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    # Ensure consistency between FACES and LABELS
    if len(FACES) != len(LABELS):
        print("Mismatch detected! Truncating to the smaller size.")
        min_samples = min(len(FACES), len(LABELS))
        FACES = FACES[:min_samples]
        LABELS = LABELS[:min_samples]

except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
    print("Error loading data. Ensure faces_data.pkl and names.pkl exist and are not corrupted.")
    exit()

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image for display
imgBackground = cv2.imread("background.png")

# Define attendance column names
COL_NAMES = ['NAME', 'TIME']

# Initialize video capture
video = cv2.VideoCapture(0)

# Set to track the last time attendance was taken for each person
last_attendance_time = {}

# Define the minimum time interval (in seconds) between attendance marks
attendance_interval = 5  # 5 seconds interval

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        # Predict the name of the detected face
        output = knn.predict(resized_img)
        predicted_name = output[0]

        # Get the current timestamp
        ts = time.time()
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

        # Check if this person has already had their attendance marked within the allowed time window
        if predicted_name not in last_attendance_time or (ts - last_attendance_time[predicted_name]) >= attendance_interval:
            exist = os.path.isfile(f"Attendance/Attendance_{date}.csv")

            # Draw rectangles and put labels on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, str(predicted_name), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

            # Record attendance
            attendance = [predicted_name, str(timestamp)]

            # Update the CSV file with the attendance
            if exist:
                with open(f"Attendance/Attendance_{date}.csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance)
            else:
                with open(f"Attendance/Attendance_{date}.csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COL_NAMES)
                    writer.writerow(attendance)

            # Update the last attendance time for this person
            last_attendance_time[predicted_name] = ts

    # Update background display
    imgBackground[162:162+480, 55:55+640] = frame
    cv2.imshow("Frame", imgBackground)

    # Key handling for actions
    k = cv2.waitKey(1)
    if k == ord('o'):
        speak("Attendance Taken.")
        time.sleep(1)

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
