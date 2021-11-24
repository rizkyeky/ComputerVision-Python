import numpy as np
import tensorflow as tf
import cv2

interpreter = tf.lite.Interpreter(model_path="coco_ssd_mobilenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def create_category_index(label_path='coco_ssd_mobilenet.txt'):
    f = open(label_path)
    category_index = {}
    for i, val in enumerate(f):
        if i != 0:
            val = val[:-1]
            if val != '???':
                category_index.update({(i-1): {'id': (i-1), 'name': val}})
            
    f.close()
    return category_index

categories = create_category_index()

cap = cv2.VideoCapture(0)

tm = cv2.TickMeter()
while True:
    
    ret, img_org = cap.read()
    tm.start()

    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    img = img.astype(np.uint8)
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    print(output_details)

    boxes = interpreter.get_tensor(output_details[0]['index'])
    labels = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num = interpreter.get_tensor(output_details[3]['index'])

    for i in range(boxes.shape[1]):
        if scores[0, i] > 0.5:
            box = boxes[0, i, :]
            x0 = int(box[1] * img_org.shape[1])
            y0 = int(box[0] * img_org.shape[0])
            x1 = int(box[3] * img_org.shape[1])
            y1 = int(box[2] * img_org.shape[0])

            cv2.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.rectangle(img_org, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)
            cv2.putText(img_org,
                categories[int(labels[0, i])]['name'],
                (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

    tm.stop()
    cv2.rectangle(img_org, (0, 0), (100, 25), (255, 255, 255), -1)
    cv2.putText(img_org, 'FPS: {}'.format(round(tm.getFPS(), 2)), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow('image', img_org)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    tm.stop()

cap.release()
cv2.destroyAllWindows()