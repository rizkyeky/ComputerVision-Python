import cv2
import numpy as np

img: np.array = cv2.imread('8.jpg')
# img = cv2.resize(img, (200,200))
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img = cv2.inRange(img, np.array([0,150, 150]), np.array([20, 255, 255]))
# print(img)

cv2.imshow("ori", img)

cv2.waitKey(0)
cv2.destroyAllWindows()