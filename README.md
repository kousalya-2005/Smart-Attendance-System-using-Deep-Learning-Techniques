## Title of the Project
A Smart Attendance System using facial recognition technology to automatically mark attendance based on student faces, streamlining the attendance process and enhancing accuracy in educational institutions.

## About
The Smart Attendance System project integrates facial recognition technology to automate the process of marking student attendance in classrooms. The system captures images of students' faces and stores them for future recognition. Using a webcam, the system detects faces in real-time and matches them against the stored dataset to mark attendance.

Key Features: 1.Face Enrollment:

2.Students capture their face images via webcam and register their name. 100 images per student are captured for improved recognition accuracy. Stored in a pickle file for fast retrieval. Automated Attendance:

3.Uses K-Nearest Neighbors (KNN) to match detected faces with the stored dataset. Marks attendance and saves the student's name with the timestamp to a CSV file. Prevents duplicate attendance marking by tracking attended students. Real-time Feedback:

4.Displays student names on the webcam feed when attendance is taken. Provides audio feedback confirming successful attendance using Windows Speech API. Streamlined Interface:

5.User-friendly interface built with Streamlit. Easy navigation between "Add Faces" and "Take Attendance" options. Accuracy and Efficiency:

6.High recognition accuracy using KNN classifier. Real-time processing for quick attendance marking with minimal delay.

7.Goal: Automates attendance marking, offering a more efficient, accurate, and scalable solution for educational institutions.

## Features
Face Enrollment: Allows students to register by capturing their face images through a webcam.

Real-Time Face Detection: Detects faces from webcam feed using Haar Cascade Classifier.

KNN-based Recognition: Uses K-Nearest Neighbors (KNN) classifier to recognize faces and mark attendance.

Automated Attendance Logging: Marks attendance when a student’s face is recognized and saves it in a CSV file.

Audio Feedback: Provides real-time audio feedback to confirm attendance marking using Windows Speech API.

Student Data Management: Stores and retrieves student face data and names using Pickle files.

Attendance History: Saves attendance records with timestamps in CSV files for each session.

User Interface: Streamlit-based interface for easy interaction and navigation between adding faces and taking attendance.

Duplicate Detection: Ensures each student’s attendance is recorded only once during a session.

Real-Time Visual Feedback: Displays student names on the video stream when their attendance is taken.

## Requirements
Software Requirements: Python (3.x) Streamlit OpenCV Scikit-learn Numpy Pickle Windows Speech API (SAPI)

Hardware Requirements: Webcam for capturing student faces A computer with sufficient processing power for real-time face detection

External Libraries: opencv-python for face detection and image processing scikit-learn for KNN classification numpy for numerical operations pickle for storing and loading data streamlit for creating the user interface

Dataset Requirements: A collection of student face images (at least 100 per student for accurate recognition) A CSV file to store attendance data

Miscellaneous: Haar Cascade Classifier XML file for face detection (haarcascade_frontalface_default.xml) A directory to save attendance logs (CSV files)

## System Architecture
<!--Embed the system architecture diagram as shown below-->

![image](https://github.com/user-attachments/assets/8ce4ec90-73cf-4631-a163-a868d767e969)


## Output

<!--Embed the Output picture at respective places as shown below as shown below-->
#### Output1 - Name of the output

![Screenshot 2023-11-25 134037](https://github.com/<<yourusername>>/Hand-Gesture-Recognition-System/assets/75235455/8c2b6b5c-5ed2-4ec4-b18e-5b6625402c16)

#### Output2 - Name of the output
![Screenshot 2023-11-25 134253](https://github.com/<<yourusername>>/Hand-Gesture-Recognition-System/assets/75235455/5e05c981-05ca-4aaa-aea2-d918dcf25cb7)

Detection Accuracy: 96.0% Note: These metrics can be customized based on your actual performance evaluations.


## Results and Impact
Automated Attendance: Reduces time and effort by eliminating manual attendance-taking.

High Accuracy: Ensures accurate attendance marking with face recognition and KNN classification.

Real-Time Feedback: Provides immediate visual and audio confirmation when attendance is marked.

Scalability: Can be easily adapted to handle larger databases and used in bigger institutions.

Improved User Experience: Streamlined, user-friendly interface for easy interaction.

Reduced Fraud: Minimizes proxy attendance by using face recognition technology.

Impact on Education: Helps educators focus more on teaching by automating attendance and providing accurate records.

This project serves as a foundation for future developments in assistive technologies and contributes to creating a more inclusive and accessible digital environment.

## Articles published / References
1.Chunduru Anilkumar; B Venkatesh; S Annapoorna, “Smart Attendance System with Face Recognition using OpenCV”, 22 September 2023
2.Jayaraj Viswanathan1, Kuralamudhan , Navaneethan and Veluchamy , “Smart Attendance System using Face Recognition”, Data Science Insights, 26 February 2024



