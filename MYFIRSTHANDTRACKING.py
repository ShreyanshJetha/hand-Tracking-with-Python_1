import cv2
import handtrackingmodule as htm
import time

ptime = 0
ctime = 0
cap = cv2.VideoCapture(0)
detector=htm.handdetector()
while True:
    success, img = cap.read()
    img=detector.findhands(img, draw=False)
    lmlist=detector.findpostion(img, draw=False)
    if len(lmlist)!=0:
        print(lmlist[4])
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 1, (255, 0, 255), 4)

    cv2.imshow("Image", img)
    # cv2.imshow("RGB", imgRGB)
    cv2.waitKey(1)
