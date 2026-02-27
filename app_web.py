import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np

st.set_page_config(page_title="BiospecklePy Web", layout="wide")

st.title("🌱 BiospecklePy - Real Time")

# --- CONFIGURAÇÃO DE REDE ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- INTERFACE ABAIXO DO VÍDEO (CONTROLES) ---
# Criamos colunas para organizar os sliders lado a lado embaixo do vídeo
col1, col2, col3, col4 = st.columns(4)

with col1:
    modo = st.radio("Modo de Visão:", ["LASCA", "CINZA"], key="modo_visao")

with col2:
    m_gray = st.slider("Filtro de Ruído", 0, 255, 20, key="slider_gray")

with col3:
    c_scale = st.slider("Contraste LASCA", 1, 100, 30, key="slider_contrast")

with col4:
    k_size = st.slider("Tamanho do Kernel", 3, 15, 5, step=2, key="slider_kernel")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Processamento base em Cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if modo == "CINZA":
        # Apenas converte de volta para BGR para exibir no player de vídeo
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        # Algoritmo LASCA
        img_f32 = gray.astype(np.float32)
        kernel = np.ones((k_size, k_size), np.float32) / (k_size**2)
        
        mean = cv2.filter2D(img_f32, -1, kernel)
        sq_mean = cv2.filter2D(img_f32**2, -1, kernel)
        std = cv2.sqrt(cv2.absdiff(sq_mean, mean**2))
        
        mean[mean == 0] = 1
        lasca = (std / mean) * (255.0 / (c_scale / 50.0))
        
        lasca_u8 = 255 - np.clip(lasca, 0, 255).astype(np.uint8)
        lasca_u8[mean < m_gray] = 0
        
        # Colorização
        result = cv2.applyColorMap(lasca_u8, cv2.COLORMAP_JET)
    
    return frame.from_ndarray(result, format="bgr24")

# --- EXIBIÇÃO DO VÍDEO ---
webrtc_streamer(
    key="biospeckle-analysis",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

st.write("---")
st.caption("BiospecklePy Web Edition - Use os controles acima para ajustar a análise.")