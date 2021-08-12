import cv2
import numpy as np
import face_recognition

newImage = face_recognition.load_image_file('Images/Adam.jpg')
newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB)
testImage = face_recognition.load_image_file('Images/Adam-test.jpg')
testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(newImage)[0]
encodeImg = face_recognition.face_encodings(newImage)[0]
cv2.rectangle(newImage, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255, 0, 255), 2)

facelocTest = face_recognition.face_locations(testImage)[0]
encodeTest = face_recognition.face_encodings(testImage)[0]
cv2.rectangle(testImage, (facelocTest[3], facelocTest[0]), (facelocTest[1], facelocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeImg], encodeTest)
faceDis = face_recognition.face_distance([encodeImg], encodeTest)
print(results, faceDis)
cv2.putText(testImage, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

cv2.imshow('Adam Grant', newImage)
cv2.imshow('Adam Test', testImage)
cv2.waitKey(0)