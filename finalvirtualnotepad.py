import cv2
import numpy as np
import os
import math
import tensorflow as tf
import HandTrackingModule as htm
import wmi
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

brushThickness = 50
eraserThickness = 100
MaxBrightness = 100
MinBrightness = 0
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

model_path = "C:/Users/srivy/Downloads/srivyshnavi.h5"
model = tf.keras.models.load_model(model_path)


folderPath = "C:/Users/srivy/Downloads/Headers"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    image = cv2.resize(image, (1280,125))
    overlayList.append(image)
header = overlayList[0]
drawColor = False 
c = wmi.WMI(namespace='wmi') 


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.5)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
brightnessControl = False
volControl = False
predictMode = False

def preprocess_image(image):
    """Preprocess the canvas image for digit prediction."""

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    resized_image = cv2.resize(gray_image, (28, 28))
    
   
    normalized_image = resized_image / 255.0
    
 
    reshaped_image = normalized_image.reshape(1, 28, 28, 1)
    

    cv2.imshow("Resized Image", resized_image)
    
    return reshaped_image



def predict_digit(image):
    """Predict the digit using the pre-trained model"""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return np.argmax(prediction)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        x0, y0 = lmList[4][1:]
        x1, y1 = lmList[8][1:]  
        x2, y2 = lmList[12][1:]
        cx, cy = (x1 + x0) // 2, (y1 + y0) 
        print(x1)
        fingers = detector.fingersUp()

        if brightnessControl:
                if fingers[1] and fingers[2] == False:
                    cv2.circle(img, (x0, y0), 15, (0, 0, 0), cv2.FILLED)
                    cv2.circle(img, (x1, y1), 15, (0, 0, 0), cv2.FILLED)
                    cv2.line(img, (x0,y0),(x1,y1), (0, 0, 0), 3)
                    cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
                    Length = math.hypot(x1-x0, y1-y0)
                    if Length<50:
                        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                    length = np.interp(Length, (50,250), (MinBrightness, MaxBrightness))
                    brightnessBar = np.interp(Length, (50, 250), (600, 250))
                    brightness = int(length) 
                    color = (0, int(255 - (brightness / MaxBrightness) * 255), int((brightness / MaxBrightness) * 255))  
                    if length <= MaxBrightness:
                        color = (0, int(255 - (brightness / MaxBrightness) * 255), int((brightness / MaxBrightness) * 255))
                        cv2.rectangle(img, (50,250), (85,600), color, 3)
                        cv2.rectangle(img, (50,int(brightnessBar)), (85,600), color, cv2.FILLED)
                        cv2.putText(img, f'{int(brightness)} %',(45,220), cv2.FONT_HERSHEY_COMPLEX,
                            1,(0, int(255 - (brightness / MaxBrightness) * 255), int((brightness / MaxBrightness) * 255)), 3)
                    c.WmiMonitorBrightnessMethods()[0].WmiSetBrightness(brightness, 0)
       
        if volControl:
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x0,y0), 15, (0, 0, 0), cv2.FILLED)
                cv2.circle(img, (x1, y1), 15, (0, 0 ,0), cv2.FILLED)
                cv2.line(img, (x0,y0),(x1,y1), (0, 0, 0), 3)
                cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
                length = math.hypot(x1-x0, y1-y0)
                vol = np.interp(length,[50,250],[minVol, maxVol])
                volBar = np.interp(length,[50, 250], [600, 250])
                volPer = np.interp(length, [50, 250], [0, 100])
                #print(int(length), vol)
                volume.SetMasterVolumeLevel(vol, None)
                if volPer <= MaxBrightness:
                    color = (0, 0, 255)
                    if length<50:
                        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                    cv2.rectangle(img, (50,250), (85,600), color, 3)
                    cv2.rectangle(img, (50,int(volBar)), (85,600), color, cv2.FILLED)
                    cv2.putText(img, f'{int(volPer)} %',(45,220), cv2.FONT_HERSHEY_COMPLEX,
                        1,color, 3)
                
       
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
            if y1 < 125:
                if 1030< x1 < 1100:
                    volControl = False
                    header = overlayList[2]
                    drawColor = (0, 0, 255)
                elif 930 < x1 < 1000:
                    volControl = False
                    header = overlayList[3]
                    drawColor = (220, 35, 35)
                elif 830 <x1 < 900:
                    volControl = False
                    header = overlayList[4]
                    drawColor = (0, 255, 0)
                elif 1130 < x1 < 1230:
                    volControl = False
                    header = overlayList[1]
                    drawColor = (255, 255, 255)
                elif 650 < x1 < 800:
                    volControl = False
                    header = overlayList[5]
                    drawColor = (0, 0, 0)
                elif 50 < x1 < 100:
                    header = overlayList[8]
                    volControl = False
                    drawColor = False
                    brightnessControl = True
                elif 150 < x1 < 250:
                    header = overlayList[7]
                    drawColor = False
                    volControl = True
                elif 330 < x1 < 460:
                    header = overlayList[6]
                    drawColor = False
                    volControl = False 
                    if not predictMode:
                        predictMode = True
                        print("Predict mode activated")
                        
                        digit = predict_digit(imgCanvas)
                        print(f'Predicted Digit: {digit}')
                        cv2.putText(img, f'Predicted Digit: {digit}', (50, 650), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                        cv2.imshow("Image", img)
                        cv2.waitKey(2000)
                        predictMode = False

       
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = header
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Save the canvas image
        cv2.imwrite('my_painting.png', imgCanvas)
        print("Canvas saved as my_painting.png")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
