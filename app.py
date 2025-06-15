import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os

# Configurar la página
st.set_page_config(page_title="Reconocimiento de Placas", layout="centered")


# Título y descripción
st.title("Parking Organization - Reconocimiento de Placas")
st.markdown("Sube una imagen con una placa vehicular y extrae su contenido usando **YOLOv8** + **TrOCR** de Microsoft para reconocimiento OCR de alta precisión.")

# Cargar modelo YOLOv8 personalizado (tu modelo entrenado)
model = YOLO("/content/drive/MyDrive/YOLOv8/entrenamientos/modelo_placas_v1/weights/best.pt")  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed").to(device)

# Función para extraer texto con TrOCR
def trocr_extract_text(image: np.ndarray) -> str:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = processor(image_rgb, return_tensors="pt").pixel_values.to(device)
    generated_ids = ocr_model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

# Carga de imagen desde el usuario
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=" Imagen Original", use_column_width=True)

    # Convertir a array de OpenCV
    img_cv = np.array(image)

    with st.spinner("Detectando placas y extrayendo texto..."):
        results = model(img_cv)
        output_img = img_cv.copy()
        textos = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                placa_crop = img_cv[y1:y2, x1:x2]

                try:
                    texto = trocr_extract_text(placa_crop)
                except:
                    texto = "[Error OCR]"

                textos.append(texto)
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_img, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        st.image(output_img, caption="📸 Resultado con texto extraído", use_column_width=True)

        if textos:
            st.success("Texto extraído:")
            for i, t in enumerate(textos, 1):
                st.markdown(f"**Placa {i}:** `{t}`")
        else:
            st.warning("No se detectó ninguna placa.")
