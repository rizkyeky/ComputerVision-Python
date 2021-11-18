import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

tm = cv2.TickMeter()

while(True):
    ret, frame = cap.read()
    tm.start()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        end_x = x + w
        end_y = y + h

        cv2.rectangle(frame, (x, y), (end_x, end_y), (0,255,0), 2)

    tm.stop()
    cv2.putText(frame, tm.getFPS(), (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    tm.reset()

cap.release()
cv2.destroyAllWindows()
