import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('../images/lena.jpg',0)

edges = cv2.Canny(img,100,200)

plt.subplot(1,2,1)
plt.imshow(img, cmap = 'gray')
plt.title('Imagen Original')
plt.xticks([])
plt.yticks([])
plt.subplot(1,2,2)
plt.imshow(edges, cmap='gray')
plt.title('Deteccion Bordes')
plt.xticks([])
plt.yticks([])

plt.show()
