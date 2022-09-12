import cv2
import numpy as np
import face_recognition
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

def findEncoding(images):
	encodelist = []
	for img in images:
		img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
		encode = face_recognition.face_encodings(img)[0]
		encodelist.append(encode)
	return encodelist

encodeListKnown = findEncoding(images)
print('Encoding Comlete')
print(encodeListKnown)
np.savetxt('venv/Data.csv', encodeListKnown)
print("da in xong")