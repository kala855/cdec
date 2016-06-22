import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np

rutaImagen = './images/rostros2.jpg'
rutaCascadaCara = './cascadas/haarcascade_frontalface_default.xml'
rutaCascadaOjo = './cascadas/haarcascade_eye.xml'

# Crear la cascada de Haar
cascadaHaar = cv2.CascadeClassifier(rutaCascadaCara)
cascadaHaarOjos = cv2.CascadeClassifier(rutaCascadaOjo)

# Lectura de la imagen
imagen = cv2.imread(rutaImagen)

imagenGris = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)

# Detectar rostros en la imagen
rostros = cascadaHaar.detectMultiScale(imagenGris,scaleFactor=1.3,minNeighbors=5)

for (x,y,w,h) in rostros:
    cv2.rectangle(imagen,(x,y),(x+w,y+h),(255,0,0),4)
    roi_gris = imagenGris[y:y+h,x:x+w]
    roi_color = imagen[y:y+h,x:x+w]
    ojos = cascadaHaarOjos.detectMultiScale(roi_gris)
    for(ox,oy,ow,oh) in ojos:
        cv2.rectangle(roi_color,(ox,oy),(ox+ow,oy+oh),(0,255,0),2)

imagen = cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)
plt.imshow(imagen)
plt.show()
