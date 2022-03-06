import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
mphands=mp.solutions.hands
hands=mphands.Hands()
mpdraw=mp.solutions.drawing_utils
ptime=0
ctime=0

while True:
    success, img = cap.read()
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)

    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:  #handlms= hand land marks for multiple hands
        for handlms in results.multi_hand_landmarks:
            for id ,lm in enumerate(handlms.landmark):
                #print(id,lm)
                height, width, channel= img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                print(id,cx,cy)
                if id ==0:
                    cv2.circle(img,(cx,cy), 25, (255,0,255), cv2.FILLED)
            mpdraw.draw_landmarks(img, handlms, mphands.HAND_CONNECTIONS)

    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_ITALIC,1,(255,0,255),4)
    cv2.imshow("Image",img)
    #cv2.imshow("RGB", imgRGB)
    cv2.waitKey(1)
