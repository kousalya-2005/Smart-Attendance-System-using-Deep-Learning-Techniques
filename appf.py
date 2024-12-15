import streamlit as st
import cv2
import numpy as np
import pickle
import os
import time
from datetime import datetime
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from win32com.client import Dispatch

# Path settings
FACE_DIR = 'data/faces_data.pkl'
NAMES_FILE = 'data/names.pkl'

# Initialize the face detector
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Function to speak (for feedback)
def speak(message):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(message)

# Load face data and labels for attendance checking
try:
    with open(NAMES_FILE, 'rb') as w:
        LABELS = pickle.load(w)
    with open(FACE_DIR, 'rb') as f:
        FACES = pickle.load(f)

    # Debugging: Print initial shapes
    print(f"Faces shape: {FACES.shape}")
    print(f"Number of labels: {len(LABELS)}")

    # Ensure consistency between FACES and LABELS
    if len(FACES) != len(LABELS):
        print("Mismatch detected! Truncating to the smaller size.")
        min_samples = min(len(FACES), len(LABELS))
        FACES = FACES[:min_samples]
        LABELS = LABELS[:min_samples]

    # Debugging: Verify shapes after truncation
    print(f"Updated Faces shape: {FACES.shape}")
    print(f"Updated number of labels: {len(LABELS)}")

except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
    st.error("Error loading data. Ensure faces_data.pkl and names.pkl exist and are not corrupted.")
    st.stop()

# Function to add faces
def add_faces():
    name = st.text_input("Enter the student's name:")
    if name:
        # Initialize video capture
        video = cv2.VideoCapture(0)
        faces_data = []
        i = 0

        st.write("Capturing faces...")

        while len(faces_data) < 100:
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                crop_img = frame[y: y + h, x: x + w, :]
                resized_img = cv2.resize(crop_img, (50, 50))

                if i % 10 == 0:
                    faces_data.append(resized_img)
                i += 1
                cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

            # Show the video stream
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) == 100:
                break

        video.release()
        cv2.destroyAllWindows()

        # Convert to numpy array and save the faces
        faces_data = np.asarray(faces_data)
        faces_data = faces_data.reshape(100, -1)

        # Update the names and face data
        try:
            if os.path.exists(NAMES_FILE):
                with open(NAMES_FILE, 'rb') as f:
                    names = pickle.load(f)
                names += [name] * 100
            else:
                names = [name] * 100
            with open(NAMES_FILE, 'wb') as f:
                pickle.dump(names, f)

            if os.path.exists(FACE_DIR):
                with open(FACE_DIR, 'rb') as f:
                    faces = pickle.load(f)
                faces = np.append(faces, faces_data, axis=0)
            else:
                faces = faces_data

            with open(FACE_DIR, 'wb') as f:
                pickle.dump(faces, f)

            st.success(f"Face data for {name} added successfully!")
        except Exception as e:
            st.error(f"Error saving data: {e}")

# Function to perform attendance
def take_attendance():
    # Train the KNN model for recognition
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    # Initialize video capture
    video = cv2.VideoCapture(0)

    # Set to track students who have had their attendance marked
    attended_names = set()

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

            # Predict the name of the detected face
            output = knn.predict(resized_img)
            predicted_name = output[0]

            # Check if attendance is already marked for this person
            if predicted_name not in attended_names:
                timestamp = datetime.now().strftime("%H:%M:%S")
                date = datetime.now().strftime("%d-%m-%Y")
                attendance = [predicted_name, timestamp]

                # Draw rectangles and put labels on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                cv2.putText(frame, predicted_name, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

                # Save attendance data to CSV file
                if os.path.isfile(f"Attendance/Attendance_{date}.csv"):
                    with open(f"Attendance/Attendance_{date}.csv", "+a") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(attendance)
                else:
                    with open(f"Attendance/Attendance_{date}.csv", "+a") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['NAME', 'TIME'])
                        writer.writerow(attendance)

                # Mark this person as attended
                attended_names.add(predicted_name)
                st.write(f"Attendance taken for {predicted_name} at {timestamp}")

        # Display the frame
        cv2.imshow("Frame", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    # Evaluation: Calculate and display accuracy
    predictions = knn.predict(FACES)
    accuracy = accuracy_score(LABELS, predictions)
    st.write(f"Attendance model accuracy: {accuracy :.2f}%")
    print(f"Attendance model accuracy: {accuracy:.2f}%")

# Streamlit layout
st.title("Smart Attendance System")

# Sidebar options for adding faces or taking attendance
option = st.sidebar.selectbox("Choose an action", ["Add Faces", "Take Attendance"])

if option == "Add Faces":
    add_faces()
elif option == "Take Attendance":
    take_attendance()
