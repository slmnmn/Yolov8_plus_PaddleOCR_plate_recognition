pip install streamlit  ultralytics Pillow torch transformers
pip install paddleocr==2.10.0
pip install paddlepaddle
pip install opencv-contrib-python


if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        OR
if texto_detectado != None :
        break


Be aware if you want to train again the model from scrath, You should download YOLOv8 (v11 didn't work to well) 


PaddleOCR instead of training our own OCR. (Did not see the necesaity to do it)