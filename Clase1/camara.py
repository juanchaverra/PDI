# Usar openCv y la cámara

import cv2
import numpy as np
import time
import sys
from matplotlib import pyplot as plt


from cv2.cv2 import VideoCapture


def get_image():
    # leer la captura
    retval, im = video.read()  # Iniciar la lectura del video
    return im



if __name__ == '__main__':

    datos = np.zeros((1,2))  # matriz con información a mostrar (diferencias en los frames)
    camara = 0
    video: VideoCapture = cv2.VideoCapture(camara)  # inicia captura de video

    fondo = None

    while True:
        img1 = get_image()
        img2 = get_image()


        if img1 is None:
            break

        #convertimos a escala de grises
        gris = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gris2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        #Aplicamos suavizado para disminuir ruido
        gris = cv2.GaussianBlur(gris, (21,21),0)
        gris2 = cv2.GaussianBlur(gris, (21,21),0)


        if fondo is None:
            fondo = gris
            continue

        #se calcula la resta entre el fondo y la imagen actual
        resta = cv2.absdiff(gris2, gris)

        #aplicando el umbral
        umbral =cv2.threshold(resta, 25,255,cv2.THRESH_BINARY)[1]

        #Dilatamos el umbral para tapar agujeros
        umbral = cv2.dilate(umbral, None, iterations=2)

        #copiamos el umbral para detectar el contorno
        contornosimg = umbral.copy()

        #buscamos contornos en la imagen
        contornos, hierarchy = cv2.findContours(contornosimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #recorremos todos los contornos encontrados
      #  for c in contornos:
            #Eliminamos los contornos más pequeños
       #     if cv2.contourArea(c) < 500:
        #        continue

        #Obtenemos los bordes del contorno, el rectangulo mayor que engloba al contorno
       # (x,y,w,h) = cv2.boundingRect(c)
        #dibujamos los rectangulos del bounds
       # cv2.rectangle(img1, (x,y), (x + w, y + h), (0 ,255,0), 2)


        cv2.imshow("camara", img1)
        cv2.imshow("resta", resta)
        cv2.imshow("umbral", umbral)


        if cv2.waitKey(10) == 27: # Espera por 'ESC'
            break



    video.release()
    cv2.destroyAllWindows()
