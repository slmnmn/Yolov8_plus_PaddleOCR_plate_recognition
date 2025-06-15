import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pytesseract
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
import re

# Configurar página
st.set_page_config(page_title="Reconocimiento de Placas Híbrido", layout="centered")
st.title("Parking Organization - OCR Híbrido")
st.markdown("Sube una imagen con una **placa vehicular** para extraer su texto usando **YOLOv8 + EasyOCR + Tesseract + TrOCR** con votación y validación.")

# Cargar modelo YOLO personalizado
model = YOLO("/content/drive/MyDrive/YOLOv8/entrenamientos/modelo_placas_v1/weights/best.pt")

# Cargar modelos OCR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed").to(device)
easyocr_reader = easyocr.Reader(['es'], gpu=torch.cuda.is_available())

# Funciones OCR
def ocr_easyocr(img):
    result = easyocr_reader.readtext(img)
    return result[0][1] if result else ''

def ocr_tesseract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    config = "--psm 7"
    return pytesseract.image_to_string(gray, config=config).strip()

def ocr_trocr(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_values = processor(rgb, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def validar_placa(text):
    return re.match(r'^[A-Z]{3}[-\s]?\d{3}$', text.replace(" ", "").replace("-", ""))

def voto_mayoria(*textos):
    votos = {}
    for texto in textos:
        if validar_placa(texto):
            key = texto.replace(" ", "").replace("-", "")
            votos[key] = votos.get(key, 0) + 1
    if votos:
        return max(votos, key=votos.get)
    return None

# Cargar imagen
file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Imagen Original", use_column_width=True)
    img_cv = np.array(image)

    with st.spinner("Detectando placa..."):
        detections = model(img_cv)
        output_img = img_cv.copy()
        encontrados = []

        for result in detections:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                placa_img = img_cv[y1:y2, x1:x2]
                st.image(placa_img, caption="Placa Detectada", use_column_width=False)

                st.markdown("### OCR por método")
                texto_easy = ocr_easyocr(placa_img)
                texto_tess = ocr_tesseract(placa_img)
                texto_trocr = ocr_trocr(placa_img)

                st.markdown(f"-EasyOCR: `{texto_easy}`")
                st.markdown(f"-Tesseract: `{texto_tess}`")
                st.markdown(f"-TrOCR: `{texto_trocr}`")

                texto_final = voto_mayoria(texto_easy, texto_tess, texto_trocr)
                if texto_final:
                    encontrados.append(texto_final)
                    st.success(f"✅ Texto validado por mayoría: `{texto_final}`")
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(output_img, texto_final, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    st.warning("⚠️ No se pudo validar una placa válida.")

        st.image(output_img, caption="Resultado Final", use_column_width=True)
        if not encontrados:
            st.info("Intenta con otra imagen o mejora el contraste de la placa.")

