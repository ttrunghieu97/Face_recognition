from sre_constants import SUCCESS
import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as np
#webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280) #chiều rộng
cap.set(4,720) #chiều cao

# Hand Detector
detecor = HandDetector(maxHands=1, detectionCon=0.8) # số hand tối đa= 1, độ tin cậy = 0.8



while True:
    # Get the frame from  the webcam
    success, img = cap.read()
    #hands
    hands,img = detecor.findHands(img) #trả ra fame ảnh nhận dữ liệu bàn tay
    # Landmark values - (x,y,z)*21 -> sumvalues = 63, rồi lưu vào 1 file để sử lý data
    
    if hands: #nếu phát hiện hands
        #get the first hand detected
        hand = hands[0]#sẽ cho ra frame hands đầu tiên
        #get the landmark list 
        lmlist = hand['lmlist']
        print(lmlist)
    cv2.imshow("Image",img)
    cv2.waitKey(1) #b1: sau do run test webcam
