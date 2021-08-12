import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# make list of images
path = 'Attendance'
images = []
Names = []
listOfImageDir = os.listdir(path)
print(listOfImageDir)
for name in listOfImageDir:
    newImage = cv2.imread(f'{path}/{name}')
    images.append(newImage)
    Names.append(os.path.splitext(name)[0])
print(Names)

# make list of encodings of images we have in our system
def findEncodings(images):
    encodedList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(img)[0]
        encodedList.append(encodeImg)
    return encodedList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            date = now.strftime('%d-%B-%Y')
            time = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{date},{time}')

encodedListOfSystem = findEncodings(images)
print('Encoding Done!')

# Capture the image
candidate = cv2.VideoCapture(0)
# to connect phone as webCam
address = "http://192.168.43.1:8080/video"
candidate.open(address)

while True:
    success, img = candidate.read()
    img = cv2.resize(img, (0,0), None, 0.25, 0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facesInFrame = face_recognition.face_locations(img)
    encodeFacesInFrame = face_recognition.face_encodings(img, facesInFrame)

    for encodeFace, faceLoc in zip(encodeFacesInFrame, facesInFrame):
        matchResult = face_recognition.compare_faces(encodedListOfSystem, encodeFace)
        matchDis = face_recognition.face_distance(encodedListOfSystem, encodeFace)
        # print(matchDis)
        matchIndex = np.argmin(matchDis)
        if matchResult[matchIndex]:
            name = Names[matchIndex]
            # print(name)
            y1,x2,y2,x1 = faceLoc
            # y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 1)
            cv2.rectangle(img, (x1,y2-8), (x2, y2), (255,0,0), cv2.FILLED)
            cv2.putText(img, name, (x1+8,y2-2), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,255,255), 1)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(2)
