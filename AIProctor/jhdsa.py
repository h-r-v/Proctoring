import cv2
import numpy as np

img = cv2.imread('Chrome-icon_8x8.png', 0)
cv2.imshow("Ico", img)

print(img)
print(img.flatten())


cv2.waitKey(0)