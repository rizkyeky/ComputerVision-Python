from imutils.video import WebcamVideoStream
import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

webcam = WebcamVideoStream(0)
webcam.start()

bbox = (0,0,0,0)
margin = 20
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

        cv2.rectangle(frame, pt1, pt2, (0,0,255), 1)
        print('crop')
    else:
        print('full')
        img = frame
        cropted = False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray)
    
    if len(faces) > 0:
        (x,y,w,h) = faces[0]    

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
