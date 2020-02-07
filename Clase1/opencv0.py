import cv2

img = cv2.imread('../Tutorial/lena.jpg', -1) #segundo argumento: 0 para escala de grises, 1 mostrar la imagen igual, -1

cv2.imshow('image', img)

cv2.waitKey(0)#la imagen queda esperando
cv2.destroyAllWindows()
cv2.imwrite('lena_cpy.png', img, -1)#la idea es realizar la copia en otro formato, pero no funcionó



