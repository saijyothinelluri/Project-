# Import necessary libraries
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Define function to find encodings of face images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Path to the directory containing student images
path = 'student'

# Load student images and their names
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(os.path.join(path, cl))
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Function to mark attendance
def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.write(f'\n{name},{dtString}')

# Define the Kivy app
class FaceRecognitionApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        self.img = Image()
        layout.add_widget(self.img)
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 fps
        return layout

    def update(self, dt):
        success, img = self.capture_webcam()
        if not success:
            return

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        # Convert OpenCV image to Kivy texture
        texture = self.cv2_to_texture(img)
        # Update Kivy image widget
        self.img.texture = texture

    def capture_webcam(self):
        cap = cv2.VideoCapture(0)
        success, img = cap.read()
        cap.release()
        return success, img

    def cv2_to_texture(self, img):
        buf = cv2.flip(img, 0).tobytes()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

# Load face recognition data
encodeListKnown = findEncodings(images)

# Run the Kivy app
if __name__ == '__main__':
    FaceRecognitionApp().run()
