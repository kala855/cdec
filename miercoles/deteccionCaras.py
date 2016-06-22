
import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np

rutaImagen = './images/rostros3.jpg'
rutaCascada = './cascadas/haarcascade_frontalcatface_extended.xml'

# Crear la cascada de Haar
cascadaHaar = cv2.CascadeClassifier(rutaCascada)

# Lectura de la imagen
imagen = cv2.imread(rutaImagen)

imagenGris = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)

# Detectar rostros en la imagen
rostros = cascadaHaar.detectMultiScale(imagenGris,scaleFactor=1.29,minNeighbors=5,minSize=(150,150),flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

print "Se han encontrado {0} rostros".format(len(rostros))

# Ahora dibujemos un rectangulo sobre los rostros
for (x,y,w,h) in rostros:
    cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,255,0),2)

imagen = cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)
plt.imshow(imagen)
plt.show()
