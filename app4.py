import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pytesseract
import torch
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
import re

# Configuración de página
st.set_page_config(page_title="Reconocimiento de Placas Híbrido + PaddleOCR", layout="centered")
st.title("Parking Organization - OCR Híbrido Mejorado")
st.markdown("Sube una imagen con una **placa vehicular** para extraer su texto usando **YOLOv8 + EasyOCR + Tesseract + TrOCR + PaddleOCR** con votación y validación.")

# Cargar modelos
model = YOLO("/content/drive/MyDrive/YOLOv8/entrenamientos/modelo_placas_v1/weights/best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed").to(device)
easyocr_reader = easyocr.Reader(['es'], gpu=torch.cuda.is_available())
paddleocr_reader = PaddleOCR(use_angle_cls=False, lang='es', use_gpu=torch.cuda.is_available())

# Subir archivo
file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

# --- Funciones OCR individuales ---
def ocr_easyocr(img):
    result = easyocr_reader.readtext(img)
    if result:
        return result[0][1]
    return ''

def ocr_tesseract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config='--psm 7')
    return text.strip()

def ocr_trocr(img):
    image = Image.fromarray(img)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = trocr_model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()

def ocr_paddleocr(img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # PaddleOCR espera BGR
    result = paddleocr_reader.ocr(img_bgr, cls=False)
    if result and len(result[0]) > 0:
        return result[0][0][1][0]
    else:
        return ''

# --- Función de votación por mayoría ---
def voto_mayoria(*textos):
    # Expresión regular de placa colombiana tipo ABC123 o ABC12D
    patron_placa = re.compile(r'\b[A-Z]{3}\d{2}[A-Z0-9]\b')

    votos = {}
    for texto in textos:
        if texto:
            matches = patron_placa.findall(texto.upper().replace(" ", "").replace("-", ""))
            for match in matches:
                votos[match] = votos.get(match, 0) + 1

    if votos:
        return max(votos, key=votos.get)
    else:
        return None

# --- Procesamiento principal ---
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
                texto_paddle = ocr_paddleocr(placa_img)

                st.markdown(f"- EasyOCR: `{texto_easy}`")
                st.markdown(f"- Tesseract: `{texto_tess}`")
                st.markdown(f"- TrOCR: `{texto_trocr}`")
                st.markdown(f"- PaddleOCR: `{texto_paddle}`")

                texto_final = voto_mayoria(texto_easy, texto_tess, texto_trocr, texto_paddle)

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
