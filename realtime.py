import cv2
import torch
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ------------------------
# Configuraciones iniciales
# ------------------------

# Carga tu modelo YOLO entrenado para detectar placas
model = YOLO("C:/Users/Amya1/Documents/Code/YOLOv8-20250605T230454Z-1-001/YOLOv8/entrenamientos/modelo_placas_v1/weights/best.pt")

# Cargar OCR PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang='es', use_gpu=torch.cuda.is_available())

# Inicializar cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

# ------------------------
# Bucle principal
# ------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Solo rota si ves que la imagen está torcida (comenta esta línea si ves bien)
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Detección con YOLO
    detections = model(frame, verbose=False)

    for result in detections:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            placa_img = frame[y1:y2, x1:x2]

            if placa_img.size == 0:
                continue  # evitar errores si el recorte falla

            try:
                # OCR Paddle
                result_ocr = ocr.ocr(placa_img, cls=False)

                if result_ocr is not None and len(result_ocr) > 0 and len(result_ocr[0]) > 0:
                    texto_detectado = result_ocr[0][0][1][0]

                    # Filtrar: solo textos de placas reales (mayores a 5 caracteres aprox.)
                    if len(texto_detectado) >= 5:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, texto_detectado, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error procesando OCR: {e}")
                continue  # sigue con el siguiente

    # Mostrar el frame procesado
    cv2.imshow('Reconocimiento de Placas en Vivo', frame)

    #  'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    """if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        OR
    if texto_detectado != None :
        break
"""

# ------------------------
# Liberar recursos
# ------------------------
cap.release()
cv2.destroyAllWindows()
print(texto_detectado)
