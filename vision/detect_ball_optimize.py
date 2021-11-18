from imutils.video import WebcamVideoStream
from imutils import grab_contours
import cv2
import numpy as np

webcam = WebcamVideoStream(0)
webcam.start()

bbox = (0,0,0,0)
margin = 30
cropted = False

tm = cv2.TickMeter()

while True:
    frame = webcam.read()

    tm.start()

    if all(bbox):
        (x, y, w, h) = bbox

        margin = 20 + int(round(w*h/10000))
        pt1 = x-margin if x-margin > 0 else 0 , y-margin if y-margin > 0 else 0
        pt2 = x+w+margin if x+w+margin < frame.shape[1] else frame.shape[1], y+h+margin if y+h+margin < frame.shape[0] else frame.shape[0]

        bbox = pt1 + pt2

        x_crop, y_crop = pt1
        x_end_crop, y_end_crop = pt2
        
        img = frame[y_crop:y_end_crop, x_crop:x_end_crop]
        cropted = True
    else:
        img = frame
        cropted = False

    imgProc = cv2.GaussianBlur(img, (9, 9), 0)
    imgProc = cv2.cvtColor(imgProc, cv2.COLOR_BGR2HSV)
    imgProc = cv2.inRange(imgProc, np.array([97, 123, 214], dtype=np.uint8), np.array([103, 225, 255], dtype=np.uint8))
    imgProc = cv2.erode(imgProc, (3, 3), iterations=2)
    imgProc = cv2.dilate(imgProc, (3, 3), iterations=2)

    cnts, _ = cv2.findContours(imgProc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = cv2.HoughCircles(imgProc, cv2.HOUGH_GRADIENT, 1.2, 100, minRadius=20, maxRadius=200)

    if len(circles) > 0:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 1)
    
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)  

        if cropted:
            x += bbox[0]
            y += bbox[1]
            
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2, 1)
        
        bbox = (x,y,w,h)

    else:
        bbox = (0,0,0,0)

    tm.stop()
    cv2.putText(frame, str(tm.getFPS()), (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    tm.reset()

cv2.destroyAllWindows()
webcam.stop()
