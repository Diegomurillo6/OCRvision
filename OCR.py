import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure, morphology
from skimage.transform import resize

## Declaración de funciones

def comparar(img):
  #img es la imagen de cada numero del 7 seg
  #Cada numero representado por un 7 segmentos tiene una cantidad fija de segmentos
  cuadrito= 0.24 #Representa un poco más de la cantidad (%) de blanco que genera
                 #cada cuadro, teniendo en cuenta que son de 4x4 y la img de 84x84
  mask = np.zeros((84, 84))
  # Img en negro de 84x84, donde se llena con una cuadro por segmento del 7 segmento
  #  -- a --
  # |       |
  # f       b
  # |       |
  # -- g --
  # |       |
  # e       c
  # |       |
  #  -- d --
  mask[18:22,  24:28] = 1 #f
  mask[18:22,  58:62] = 1 #b
  mask[58:62,  22:26] = 1 #e
  mask[58:62,  58:62] = 1 #c
  mask[4:8,  32:36] = 1 #a
  mask[38:42,  38:42] = 1 #d
  mask[72:76,  38:42] = 1 #g

  comparador_1= mask * img # AND de la mascara creada y la imagen

  # Si el % de blanco de la imagen es menor al porcentaje de 2 cuadros
  if comparador_1.mean() * 100 < cuadrito*2:
    #El numero 1 tiene 2 segmentos
      return 1
  elif comparador_1.mean() * 100 < cuadrito*3:
    #El numero 7 tiene 3 segmentos
      return 7
  elif comparador_1.mean() * 100 < cuadrito*4:
    #El numero 4 tiene 4 segmentos
      return 4
  elif comparador_1.mean() * 100 < cuadrito*5:
    #Los numeros 2, 3 y 5 tienen 5 segmentos
      return evaluar2(img)
      #Se pasa la img a la funcion evaluar2 para diferenciar cada numero
  elif comparador_1.mean() * 100 < cuadrito*6:
    #Los numeros 0, 6 y 9 tienen 6 segmentos
      return evaluar3(img)
      #Se pasa la img a la funcion evaluar3 para diferenciar cada numero
  elif comparador_1.mean() * 100 < cuadrito*7:
    #El numero 8 tiene 7 segmentos
      return 8
  return None # En caso de error


def evaluar2(img,tol=1e-3):
  #img es la imagen de cada numero del 7 seg
  #tol es la tolerancia, tecnicamente deberia ser 0 pero como factor de seguridad
  #Mascara b, img 84 x84 con un cuadro blanco en la posición del segmento b
  mask_b = np.zeros((84, 84))
  mask_b[18:22,  58:62] = 1 #b
  #Mascara c, img 84 x84 con un cuadro blanco en la posición del segmento c
  mask_e = np.zeros((84, 84))
  mask_e[58:62,  22:26] = 1 #e

  # Al hacer un AND de la imagen y la mascara hay un elmento mayor a la tolerancia
  if not np.all(np.abs(img * mask_e) <= tol):
    #Si se tiene un segmento en en la posicion e el numero es 2
    return 2
  elif not np.all(np.abs(img * mask_b) <= tol):
    #Si se tiene un segmento en en la posicion b el numero es 3
    #(esto tambien sucede con el 2, por eso el 2 se evalua primero)
    return 3
  else:
    #En cualquier otro caso el numero es 5
    return 5

def evaluar3(img,tol=1e-3):
  #img es la imagen de cada numero del 7 seg
  #tol es la tolerancia, tecnicamente deberia ser 0 pero como factor de seguridad
  #Mascara b, img 84 x84 con un cuadro blanco en la posición del segmento b
  mask_b = np.zeros((84, 84))
  mask_b[18:22,  58:62] = 1 #b
  #Mascara c, img 84 x84 con un cuadro blanco en la posición del segmento c
  mask_e = np.zeros((84, 84))
  mask_e[58:62,  22:26] = 1 #e

  #Al hacer un AND de la imagen y la mascara no hay un elmento mayor a la tolerancia
  if np.all(np.abs(img * mask_e) <= tol):
    #Si no se tiene un segmento en en la posicion e el numero es 9
      return 9
  elif np.all(np.abs(img * mask_b) <= tol):
    #Si no se tiene un segmento en en la posicion b el numero es 6
      return 6
  else:
    #En cualquier otro caso el numero es 5
      return 0

def formato(datos):
  #Recibe los numeros obtenidos al evaluar cada numero de la imagen tomada
  # Se converten a un string
  s = str(datos)
  #Detercta si empieza con ":"
  es_pm = s.startswith("dot")
  #Se remplazan los ":" existentes
  s = s.replace("dot", "")
  #Rellenar con ceros a la izquierda hasta 4 dígitos
  s = s.zfill(4)
  #Separar en horas y minutos (son 4 digitos)
  horas = s[:2]
  minutos = s[2:]
  #Formado horas:minutos
  resultado = f"{horas}:{minutos}"
  #Se añade pm o no si tiene un punto al inicio
  if es_pm:
      resultado += " pm"

  return resultado

def clasificar(img):
  #Dependiendo el porcentaje de blanco de la imagen se determina si es un punto
  #que separa las horas o un numero
  porcent = img.mean() * 100
  if porcent > 60:
      return 'dot'
  else:
    return comparar(img)

def binarizar(img):
  #Apartir de la imagen original que toma la camara
  imagen = color.rgb2gray(img)  #Se pasa de color a escala de grises
  thresh = filters.threshold_otsu(imagen) #Se obtiene un limite con un filtro otsu automatico
  binary = imagen > thresh # Se binariza segun el limite otsu
  return binary

def recortar(img):
  # De la imagen original tomada por la camara
  image = binarizar(img) # se binariza
  #Eliminan elementos pequeños que podrian quedar sueltos de la binarizacion
  binary1 = morphology.remove_small_objects(image, min_size=50)
  #Se expanden las partes blancas de la imagen
  binary1 = morphology.dilation(image, morphology.square(5))

  #Se etiqueta cada region identificada de la imagen
  labels = measure.label(binary1)
  regiones = measure.regionprops(labels)

  #Se ordenan las regiones encontradas de izquierda a derecha
  regiones = sorted(regiones, key=lambda r: r.bbox[1])
  #Tamaño de la imagen (84x84) de cada numero
  SIZE = 84
  #Lista para cada objeto (img de cada numero)
  objetos = []

  #Por cada region
  for region in regiones:
    #Se toman en cuenta solo las regiones mayores a 100
      if region.area < 100:
          continue

      #Se definen los limites de cada region detectada, para luego recortarlos
      minr, minc, maxr, maxc = region.bbox
      #Se extre la region de interes de la imagen
      roi = binary1[minr:maxr, minc:maxc]

      #Se obtienen las dimensiones de la region de interes
      h, w = roi.shape
      #La escala se saca del lado mayor
      escala = SIZE / max(h, w)
      #Se calculan las nuevas dimensiones manteniendo la relación de aspecto
      new_h = int(h * escala)
      new_w = int(w * escala)
      #Se redimensiona la roi
      roi_resized = resize(roi, (new_h, new_w), anti_aliasing=False)

      #Se crea una imagen vacia de 84x84
      canvas = np.zeros((SIZE, SIZE))
      #Se calculan posiciones iniciales para centrar la imagen
      start_y = (SIZE - new_h) // 2
      start_x = (SIZE - new_w) // 2

      #Se coloca la roi redimencionado en la imagen creada
      canvas[start_y:start_y+new_h, start_x:start_x+new_w] = roi_resized
      #Se guardan los resultados en al lista
      objetos.append(canvas)

  #variable resultados como un str vacio
  resultado = ""
  #por cada objeto en objetos (cada imagen de un numero sacado de la imagen original)
  for i, obj in enumerate(objetos):
      #Se clasifica cada imagen de un numero
      clasificacion = clasificar(obj)
      #Se va llenando resultado con cada numero/punto en orden
      resultado += str(clasificacion)

  #Se imprime el resultado luego de darle formato
  return print(formato(resultado))

### Inicio

#Se captura la camara 1, ya que la 0 es la incluida por la laptop
cap = cv2.VideoCapture(1)

#En caso de no porder abrir la camara se impirme un error
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

print("Presiona 's' para capturar o 'q' para salir")

#Variable para guardar la foto
foto = None

while True:
    lectura, frame = cap.read()

    if not lectura:
        print("Error al capturar imagen")
        break
    #Se crea una copia de lo que ve la camara
    frame_limpio = frame.copy()

    #Se obtenienen las dimenciones de la foto
    h, w = frame.shape[:2]
    #Se dibuja una línea vertical pasando por el centro
    cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
    ##Se dibuja una línea horizontal pasando por el centro
    cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 0), 1)
    #Se muestran las lineas en la camara (estas no aparecen en la foto a analizar)
    cv2.imshow("Camara", frame)

    #Se guarda el codigo ASCII de la teclala tecla presionada
    key = cv2.waitKey(1) & 0xFF
    #Si se presiona s se analiza el frame de la imagen que tiene la camara
    #la imagen como tal no se guarda sino que se trata como una variable
    if key == ord('s'):
        foto = frame_limpio
        print("La hora es:")
        recortar(foto) #Llamado a la funcion para analizar la imagen
    #Si se presiona q se termina el programa
    elif key == ord('q'):
        break

#Se libera la camara y se cierran las ventanas creadas
cap.release()
cv2.destroyAllWindows()