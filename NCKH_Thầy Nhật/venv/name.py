import cv2
import numpy as np
import os


path = 'ImagesAttendace'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

with open('venv/name.txt', 'w') as wf:
    for text in classNames:
        wf.write(text + ',')