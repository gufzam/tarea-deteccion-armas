import cv2
from ultralytics import YOLO
from playsound3 import playsound
import threading
import time
import os

modelo_general = YOLO('yolov8n.pt')
RUTA_ARMAS = 'armas.pt'
modelo_armas = YOLO(RUTA_ARMAS) if os.path.exists(RUTA_ARMAS) else None

CLASE_PERSONA = 0
CLASE_MOTO = 3
ultima_alarma_moto = 0
ultima_alarma_arma = 0
INTERVALO_ALARMA = 3.0

def reproducir_sonido(archivo):
    try:
        if os.path.exists(archivo):
            playsound(archivo)
        else:
            print(f'[AVISO] No existe: {archivo}')
    except Exception as e:
        print(f'Error audio {archivo}: {e}')

def verificar_dos_en_moto(cajas_personas, cajas_motos):
    for moto in cajas_motos:
        mx1, my1, mx2, my2 = moto
        margen_x = (mx2 - mx1) * 0.2
        margen_y = (my2 - my1) * 0.2
        area = (mx1 - margen_x, my1 - margen_y, mx2 + margen_x, my2 + margen_y)
        personas_en_moto = 0
        for persona in cajas_personas:
            px1, py1, px2, py2 = persona
            cx = (px1 + px2) / 2
            cy = (py1 + py2) / 2
            if area[0] < cx < area[2] and area[1] < cy < area[3]:
                personas_en_moto += 1
        if personas_en_moto >= 2:
            return True
    return False

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('ERROR: No se pudo abrir la camara.')
    exit()

print("Presiona 'q' para salir.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    resultados_gen = modelo_general(frame, verbose=False)[0]
    cajas_personas, cajas_motos = [], []
    for box in resultados_gen.boxes:
        cls_id = int(box.cls[0])
        coords = box.xyxy[0].tolist()
        if cls_id == CLASE_PERSONA:
            cajas_personas.append(coords)
            cv2.rectangle(frame, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)
        elif cls_id == CLASE_MOTO:
            cajas_motos.append(coords)
            cv2.rectangle(frame, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (255, 0, 0), 2)

    arma_detectada = False
    if modelo_armas is not None:
        resultados_armas = modelo_armas(frame, verbose=False)[0]
        for box in resultados_armas.boxes:
            if float(box.conf[0]) < 0.5:
                continue
            coords = box.xyxy[0].tolist()
            cv2.rectangle(frame, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 0, 255), 3)
            cv2.putText(frame, 'ARMA', (int(coords[0]), int(coords[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            arma_detectada = True

    ahora = time.time()
    if arma_detectada:
        if ahora - ultima_alarma_arma > INTERVALO_ALARMA:
            print('ALERTA MAXIMA: ARMA DETECTADA!')
            threading.Thread(target=reproducir_sonido, args=('alarma_arma.mp3',), daemon=True).start()
            ultima_alarma_arma = ahora
    elif verificar_dos_en_moto(cajas_personas, cajas_motos):
        if ahora - ultima_alarma_moto > INTERVALO_ALARMA:
            print('ALERTA: Dos personas en moto detectadas.')
            threading.Thread(target=reproducir_sonido, args=('alarma_moto.mp3',), daemon=True).start()
            ultima_alarma_moto = ahora

    cv2.imshow('Detector de Seguridad', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
