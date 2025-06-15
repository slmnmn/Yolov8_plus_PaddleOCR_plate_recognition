import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr
import torch
import tempfile
import os

# Configurar la página
st.set_page_config(page_title="Reconocimiento de Placas", layout="centered")

# Título y descripción
st.title("Parking Organization - Reconocimiento de Placas")
st.markdown(
    "Sube una imagen con una placa vehicular. Usamos **YOLOv8** para detectar la placa y **EasyOCR** para leer los caracteres. Mostramos cada paso del proceso visualmente."
)

# Cargar modelo YOLOv8 personalizado (ajusta la ruta si es necesario)
model_path = "/content/drive/MyDrive/YOLOv8/entrenamientos/modelo_placas_v1/weights/best.pt"
model = YOLO(model_path)

# Cargar EasyOCR
reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

# Función de preprocesamiento
def preprocesar_imagen(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st.image(gray, caption="Paso 1: Escala de grises", use_column_width=True)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    st.image(blurred, caption="Paso 2: Suavizado", use_column_width=True)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 15
    )
    st.image(thresh, caption="Paso 3: Umbral adaptativo", use_column_width=True)

    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    st.image(morphed, caption="Paso 4: Cierre morfológico", use_column_width=True)

    return morphed

# Cargar imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen original", use_column_width=True)
    img_cv = np.array(image)

    with st.spinner("Detectando placas y extrayendo texto..."):
        resultados = model(img_cv)
        output_img = img_cv.copy()
        textos_detectados = []

        for resultado in resultados:
            for box in resultado.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                placa = img_cv[y1:y2, x1:x2]

                st.markdown("---")
                st.image(placa, caption="Placa detectada (recortada)", use_column_width=True)

                preprocesada = preprocesar_imagen(placa)

                ocr_result = reader.readtext(preprocesada, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                texto_filtrado = ""

                for (bbox, text, conf) in ocr_result:
                    text_clean = text.replace(" ", "").upper()
                    if 5 <= len(text_clean) <= 7 and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" for c in text_clean):
                        texto_filtrado = text_clean
                        break  # Solo tomamos el primer texto que parezca una placa válida

                if texto_filtrado:
                    texto = texto_filtrado
                else:
                    texto = "[No detectado]"

                textos_detectados.append(texto)

                # Dibujar caja y texto
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_img, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        st.markdown("---")
        st.image(output_img, caption="Resultado final con textos extraídos", use_column_width=True)

        if textos_detectados:
            st.success("Texto de placa detectado:")
            for i, t in enumerate(textos_detectados, 1):
                st.markdown(f"**Placa {i}:** `{t}`")
        else:
            st.warning("No se detectaron textos válidos.")
