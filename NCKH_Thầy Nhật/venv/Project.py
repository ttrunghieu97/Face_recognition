from unittest import findTestCases
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import name

data = np.loadtxt("venv/Data.csv")
name = ()
def markAttendace(name):
    with open('venv/Attendace.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

img = face_recognition.load_image_file('ImagesAttendace/HoangLV.jpg')
#cap = cv2.VideoCapture(0)
while True:
    #success, img = cap.read()
    imgS = cv2.resize(img,(0,0), None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_RGB2BGR)
        
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(data,encodeFace)
        faceDis = face_recognition.face_distance(data,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = name.classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            cv2.putText(img,"day la : ",(1,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.putText(img,name,(120,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            markAttendace(name)
    cv2.imshow('Webcam',img)
    cv2.waitKey(0)


