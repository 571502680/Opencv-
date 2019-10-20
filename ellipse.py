import cv2
import numpy as np

img = np.zeros((512,512,3),np.uint8)
cv2.ellipse(img,(260,240),(170,290),0,0,360,(255,255,255),3)
cv2.imshow("test",img)
cv2.waitKey(0)
cv2.destroyAllWindows()