import serial
import tensorflow as tf
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# Cargar el modelo preentrenado
modelo = load_model('C:/Users/Usuario/Desktop/TESIS/codigos_py/model_saved/Prueba_1/modelo_prueba_1.h5')
print('------------------------------------------------')
print('-------------- MODELO SUBIDO -------------------')
# Cargar los pesos del modelo
modelo.load_weights('C:/Users/Usuario/Desktop/TESIS/codigos_py/model_saved/Prueba_1/pesos_prueba_1.h5')
print('------------------------------------------------')
print('-------------- PESOS SUBIDO --------------------')

# Umbral de probabilidad
umbral_obstaculo = 0.90
umbral_sin_obstaculo = 0.10
class_names = ['Sin Obstaculos', 'Obstaculos']

# Comunicacion con ARDUINO
arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)  # Puerto, los Baudios y tiempo de espera

# Funcion para realizar la deteccion y dibujar el rectangulo
def detectar_objeto(frame):
    # Redimensionar la imagen a las dimensiones con las que fue entrenado el modelo
    imagen = cv2.resize(frame, (150, 150))  # REDIMENSIONAR
    imagen = np.expand_dims(imagen, axis=0)
    # Realizar la prediccion
    print('------------------------------------------------------')
    predicciones = modelo.predict(imagen)  # [[0.]]
    # Esta prediccion genera una salida de 2 dimensiones
    print(predicciones)
    respuesta = predicciones[0]  # Toma la dimension 0
    print(respuesta)

    # Obtener la clase predicha
    if respuesta <= umbral_sin_obstaculo:
        arduino.write(b'N')  # b = byte
        print(class_names[0])
        h, w, _ = frame.shape
        color = (0, 0, 255)  # Color del rectangulo en formato BGR (ROJO en este caso)
        espesor = 4  # Grosor del rectangulo
        # Dibujar el rectangulo en la imagen
        cv2.rectangle(frame, (0, 0), (w, h), color, espesor)
    elif np.sum(predicciones) >= umbral_obstaculo:
        arduino.write(b'P')  # b = byte
        print('Obstaculo')
        h, w, _ = frame.shape
        color = (0, 255, 0)  # Color del rectangulo en formato BGR (VERDE en este caso)
        espesor = 4  # Grosor del rectangulo
        # Dibujar el rectangulo en la imagen
        cv2.rectangle(frame, (0, 0), (w, h), color, espesor)
    else:
        arduino.write(b'C')  # b = byte
        print('Confusion')
        h, w, _ = frame.shape
        color = (255, 0, 0)  # Color del rectangulo en formato BGR (AZUL en este caso)
        espesor = 4  # Grosor del rectangulo
        # Dibujar el rectangulo en la imagen
        cv2.rectangle(frame, (0, 0), (w, h), color, espesor)
    return respuesta

# Configurar la captura de video desde la camara
capture = cv2.VideoCapture(0)
print('----------------------------------------------')
print('------------- Camara ACTIVADA ----------------')
print('----------------------------------------------')
# Bucle principal
while capture.isOpened():
    # Capturar un cuadro de la camara
    ret, frame = capture.read()
    if ret:
        # Realizar la deteccion en el cuadro capturado
        detectar_objeto(frame)
        # Mostrar la imagen resultante
        cv2.imshow('Deteccion de Obstaculo', frame)
        # Esperar 1 milisegundo
        key = cv2.waitKey(1) & 0xFF
        # Salir del bucle si se presiona la tecla 's'
        if key == ord('s'):
            break
    else:
        break

# Liberar la captura de video y cerrar todas las ventanas
capture.release()
cv2.destroyAllWindows()
arduino.close()
