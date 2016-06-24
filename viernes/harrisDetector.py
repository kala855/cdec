import cv2
from matplotlib import pyplot as plt
import numpy as np

nombreImagen = '../images/fotoGrupal.jpg'

img = cv2.imread(nombreImagen)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

dst = cv2.dilate(dst,None)

img[dst>0.01*dst.max()]=[0,0,255]

#cv2.imshow('dst',img)

img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img2)
plt.show()

#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()
