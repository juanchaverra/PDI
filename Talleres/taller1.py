import cv2

import numpy as np

"""
def mouse(event, x, y, flags, imagen): #Posición del mouse
    if event==cv2.EVENT_MOUSEMOVE:
        print('RGB Pixel (', x , ', ', y ,'): ',
        imagen.item(y,x,2), '/', imagen.item(y,x,1), '/',
        imagen.item(y,x,0))
"""


# 0 para azul, 1 para verde, 2 para rojo

def Separar(imagen):
    while 1:

        imagenGrayScale = cv2.imread("rgb.png", 0)
        cv2.namedWindow('imageGray')
        #  cv2.setMouseCallback('imageGray', mouse)
        cv2.imshow('imageGray', imagenGrayScale)

        cv2.namedWindow('image')
        #       cv2.setMouseCallback('image', mouse)
        cv2.imshow('image', imagen)

        cv2.namedWindow('blue')
        #   cv2.setMouseCallback('b',mouse)
        b = imagen.copy()
       # b[:, :, 1] = 0  # cero verde
#        b[:, :, 2] = 0  # cero rojo
        cv2.imshow('blue', b)

        cv2.namedWindow('red')
        r = imagen.copy()
        r[:, :, 0] = 0
        r[:, :, 1] = 0
        cv2.imshow('red', r)

        cv2.namedWindow('green')
        g = imagen.copy()
        g[:, :, 0] = 0
        g[:, :, 2] = 0
        cv2.imshow('green', g)

        negro = g.copy()
        negro[:, :, 1] = 0
        cv2.namedWindow('game')

        for i in range(5):
            cv2.imshow('game', negro)
            cv2.waitKey(1000)
            cv2.imshow('game', g)
            cv2.waitKey(1000)
            gr = g + r
            cv2.imshow('game', gr)
            cv2.waitKey(1000)
            grb = gr + b
            cv2.imshow('game', grb)
            cv2.waitKey(1000)

        k = cv2.waitKey(0)
        if k == ord('s'):
            break
    cv2.destroyAllWindows()


def contraste(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def Intensidad(img):
    while 1:
        # Imagen normal
        cv2.namedWindow("Imagen")

        # Escala de gris bajo contrast
        gray = cv2.imread("img.jpg", 0)
        gray = cv2.resize(gray, (250, 250))

        # Mas contraste
        gm1 = contraste(img, 0.75)
        gm2 = contraste(img, 0.5)
        gm3 = contraste(img, 0.25)

        # Contraste para gris
        gris1 = contraste(gray, 0.75)
        gris2 = contraste(gray, 0.5)
        gris3 = contraste(gray, 0.25)

        # Grafica de las cuatro imagenes cambiando gamma
        img3 = cv2.hconcat([img, gm1])
        img4 = cv2.hconcat([gm2, gm3])
        img5 = cv2.vconcat([img3, img4])

        # images of color gray
        img6 = cv2.hconcat([gray, gris1])
        img7 = cv2.hconcat([gris2, gris3])
        img8 = cv2.vconcat([img6, img7])

        # Graph the all image
        cv2.imshow("Imagen", img5)
        cv2.imshow("Scale gray", img8)

        y = cv2.waitKey(0)
        if y == ord('z'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    imagen = cv2.imread("car.jpeg", 1)
    print(imagen.shape)
    #    imagen = cv2.resize(imagen,(200,200))
    img = cv2.imread("paisaje.jpeg")
    img = cv2.resize(img, (250, 250))
    imagen = cv2.resize(imagen, (250, 250))
    #Intensidad(imagen)
    #Separar(imagen)
    print(imagen.shape)
    cv2.imshow('Prueba', imagen)
    otr = imagen.copy() + 10
   # otr = otr + 100
    #otr[:, :, 0] = 0
    cv2.imshow('otra', imagen)
    imagen = imagen
    img = img
    suma = contraste(imagen, 0.5)
    #suma = imagen/2 + img/2
    cv2.imshow('suma', suma)
    cv2.imshow('sur', imagen)

    cv2.waitKey(0)
    # f = np.reshape(e, 720)

