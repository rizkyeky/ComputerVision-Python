import cv2

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

cap = cv2.VideoCapture(0)

h = s = v = xpos = ypos = 0

def draw_function(event, x, y, flags, frame):
    if event == cv2.EVENT_LBUTTONUP:
        global b, g, r, xpos, ypos, clicked
        
        xpos = x
        ypos = y
        
        imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = imgHSV[y, x]
        
        print(h,s,v)
        
cv2.namedWindow('image')

while(True):

    ret, ori = cap.read()

    cv2.setMouseCallback('image', draw_function, ori)
    cv2.imshow('image', ori)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
