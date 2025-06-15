import streamlit as st
import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
from PIL import Image, ImageOps
import torch

# Configuración
st.set_page_config(page_title="OCR de Placas", layout="centered")
st.title("🚗 OCR de Placas - YOLO + PaddleOCR")

# Cargar modelos
model_yolo = YOLO("C:/Users/Amya1/Documents/Code/YOLOv8-20250605T230454Z-1-001/YOLOv8/entrenamientos/modelo_placas_v1/weights/best.pt") #Remplace a modelo
ocr_model = PaddleOCR(use_angle_cls=False, lang='es', use_gpu=torch.cuda.is_available())

# Función para corregir orientación usando EXIF
def corregir_orientacion(img_pil):
    return ImageOps.exif_transpose(img_pil)

# Función de detección de placa
def detectar_placa(img_cv):
    results = model_yolo.predict(img_cv, imgsz=640, conf=0.5)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0].astype(int)
        placa = img_cv[y1:y2, x1:x2]
        return placa, (x1, y1, x2, y2)
    else:
        return None, None

# Función OCR con PaddleOCR
def ocr_paddleocr(img_cv):
    img_bgr = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    result = ocr_model.ocr(img_bgr, cls=False)
    textos_detectados = []
    if result and len(result[0]) > 0:
        for linea in result[0]:
            textos_detectados.append(linea[1][0])
    return textos_detectados

# Subir imagen
file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if file:
    # Abrir imagen
    image_pil = Image.open(file).convert("RGB")
    # Corregir orientación usando EXIF
    image_pil = corregir_orientacion(image_pil)
    img_cv = np.array(image_pil)

    st.subheader("🖼️ Imagen Original Corregida")
    st.image(image_pil, caption="Imagen Corregida", use_column_width=True)

    # Detección
    with st.spinner("Detectando placa con YOLO..."):
        placa_recortada, box = detectar_placa(img_cv)

    if placa_recortada is not None:
        # Dibujar el cuadro en la imagen
        img_con_cuadro = img_cv.copy()
        x1, y1, x2, y2 = box
        cv2.rectangle(img_con_cuadro, (x1, y1), (x2, y2), (0, 255, 0), 3)

        st.subheader("🔲 Imagen con Placa Detectada")
        st.image(img_con_cuadro, caption="Detección de Placa", channels="RGB", use_column_width=True)

        st.subheader("🔎 Recorte de la Placa")
        st.image(placa_recortada, caption="Placa Recortada", use_column_width=False)

        # OCR
        with st.spinner("Leyendo caracteres con PaddleOCR..."):
            textos = ocr_paddleocr(placa_recortada)

        st.subheader("📝 Texto Detectado")
        if textos:
            for i, texto in enumerate(textos, 1):
                st.markdown(f"- {i}. **`{texto}`**")
        else:
            st.warning("⚠️ No se detectó texto en la placa.")
    else:
        st.error("❌ No se detectó ninguna placa en la imagen.")
