import tensorflow_hub as hub
import cv2
import tensorflow as tf
from pandas import read_csv

detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
labels = read_csv('labels.csv',sep=';',index_col='ID')
labels = labels['OBJECT (2017 REL.)']

cap = cv2.VideoCapture(0)

tm = cv2.TickMeter()
while(True):
    #Capture frame-by-frame
    ret, frame = cap.read()
    tm.start()

    #Convert img to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Is optional but i recommend (float convertion and convert img to tensor image)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

    #Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    
    boxes, scores, classes, num_detections = detector(rgb_tensor)
    
    pred_labels = classes.numpy().astype('int')[0]
    
    pred_labels = [labels[i] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]
   
   #loop throughout the detections and place a box around it  

    for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5:
            continue
            
        score_txt = f'{100 * round(score,0)}'
        frame = cv2.rectangle(frame, (xmin, ymax),(xmax, ymin),(0,255,0),1)      
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,label,(xmin, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame,score_txt,(xmax, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)

    tm.stop()
    cv2.putText(frame, 'FPS: {}'.format(tm.getFPS()), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    #Display the resulting frame
    cv2.imshow('black and white',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    tm.reset()

cap.release()
cv2.destroyAllWindows()