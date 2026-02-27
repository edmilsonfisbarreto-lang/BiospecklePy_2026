import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np

st.set_page_config(page_title="BiospecklePy Web", layout="wide")

st.title("🌱 BiospecklePy - Real Time")

# Controles na Barra Lateral
st.sidebar.header("Configurações")
min_gray = st.sidebar.slider("Filtro de Ruído", 0, 255, 20)
contrast = st.sidebar.slider("Contraste LASCA", 1, 100, 30)

def video_frame_callback(frame):
    # 1. Converte o frame da câmera para array
    img = frame.to_ndarray(format="bgr24")
    
    # 2. Processamento Cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_f32 = gray.astype(np.float32)

    # 3. Lógica LASCA (Simplificada para fluidez na web)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    
    mean = cv2.filter2D(img_f32, -1, kernel)
    sq_mean = cv2.filter2D(img_f32**2, -1, kernel)
    std = cv2.sqrt(cv2.absdiff(sq_mean, mean**2))
    
    # Evita divisão por zero e aplica máscara
    mean[mean == 0] = 1
    lasca = (std / mean) * (255.0 / (contrast / 50.0))
    
    # 4. Finalização
    lasca_u8 = 255 - np.clip(lasca, 0, 255).astype(np.uint8)
    lasca_u8[mean < min_gray] = 0
    
    # Aplica Mapa de Cores
    result = cv2.applyColorMap(lasca_u8, cv2.COLORMAP_JET)
    
    return frame.from_ndarray(result, format="bgr24")

# Componente de vídeo corrigido
webrtc_streamer(
    key="biospeckle-analysis",
    video_frame_callback=video_frame_callback,
    rtc_configuration={ 
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

st.info("Clique em 'START' acima e autorize a câmera no seu navegador.")