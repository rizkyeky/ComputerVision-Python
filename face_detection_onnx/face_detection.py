import numpy as np
import cv2

def visualize(input, faces, thickness=2):
    output = input.copy()
    if faces[1] is not None:
        for face in faces[1]:
            coords = face[:-1].astype(np.int32)
            cv2.rectangle(output, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
            cv2.circle(output, (coords[4], coords[5]), 2, (255, 0, 0), 2)
            cv2.circle(output, (coords[6], coords[7]), 2, (0, 0, 255), 2)
            cv2.circle(output, (coords[8], coords[9]), 2, (0, 255, 0), 2)
            cv2.circle(output, (coords[10], coords[11]), 2, (255, 0, 255), 2)
            cv2.circle(output, (coords[12], coords[13]), 2, (0, 255, 255), 2)
    return output

if __name__ == '__main__':

    detector = cv2.FaceDetectorYN.create(
        'yunet.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )

    cap = cv2.VideoCapture(0)
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    tm = cv2.TickMeter()
    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        tm.start()
        faces = detector.detect(frame)
        frame = visualize(frame, faces)
        tm.stop()

        cv2.rectangle(frame, (0, 0), (100, 25), (255, 255, 255), -1)
        cv2.putText(frame, 'FPS: {}'.format(round(tm.getFPS(), 2)), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow('Cam', frame)

        tm.reset()
        