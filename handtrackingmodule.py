import cv2
import mediapipe as mp
import time

class handdetector():
    def __init__(self,mode=False, maxhands=2,modelcomplex=1,detection_confidence=0.5, track_confidence=0.5):
        self.mode=mode
        self.maxhands=maxhands
        self.modelcomplex=modelcomplex
        self.detection_confidence=detection_confidence
        self.track_confidence=track_confidence


        self.mphands=mp.solutions.hands
        self.hands=self.mphands.Hands(self.mode,self.maxhands,self.modelcomplex,self.detection_confidence,self.track_confidence)
        self.mpdraw=mp.solutions.drawing_utils

    def findhands(self, img, draw=True):
        imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)

        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:  #handlms= hand land marks for multiple hands
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handlms, self.mphands.HAND_CONNECTIONS)
        return img

    def findpostion(self, img, handNo=0, draw=True):
        lmlist=[]
        if self.results.multi_hand_landmarks:
            myhand=self.results.multi_hand_landmarks[handNo]

            for id ,lm in enumerate(myhand.landmark):
                #print(id,lm)
                height, width, channel= img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                #print(id,cx,cy)
                lmlist.append([id, cx, cy])
                if draw and id==0:
                    cv2.circle(img,(cx,cy), 25, (255,0,255),cv2.FILLED)
        return lmlist




def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    detector=handdetector()
    while True:
        success, img = cap.read()
        img=detector.findhands(img)
        lmlist=detector.findpostion(img)
        if len(lmlist)!=0:
            print(lmlist[4])
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 1, (255, 0, 255), 4)

        cv2.imshow("Image", img)
        # cv2.imshow("RGB", imgRGB)
        cv2.waitKey(1)

if __name__== "__main__":
    main()