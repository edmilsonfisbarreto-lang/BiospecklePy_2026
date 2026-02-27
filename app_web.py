import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np

# Configuração da página
st.set_page_config(page_title="BiospecklePy", layout="wide")

# CSS para Estilo Verde e Posicionamento da Escala de Cores
st.markdown("""
    <style>
    /* Estilo dos Sliders */
    .stSlider [data-baseweb="slider"] [role="slider"] { background-color: #2D5A27; }
    .stSlider [data-baseweb="slider"] [aria-valuemax] { background-color: #2D5A27; }
    
    /* Container do Vídeo e Escala */
    .video-flex-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        margin-bottom: 20px;
    }
    .color-bar {
        width: 30px;
        height: 350px;
        background: linear-gradient(to top, #00008F, #0000FF, #00FFFF, #FFFF00, #FF0000, #800000);
        border: 2px solid #555;
        border-radius: 5px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        color: white;
        font-weight: bold;
        font-size: 10px;
        font-family: sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 BiospecklePy")

# --- CONTROLES UNIFICADOS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    modo = st.radio("Modo de Visão:", ["LASCA", "CINZA"])
    escolha_camera = st.selectbox("Escolher Câmera:", ["user", "environment"], 
                                 format_func=lambda x: "Câmera Frontal/Padrão" if x == "user" else "Câmera Externa/Traseira")

with col2:
    m_gray = st.slider("Filtro de Ruído", 0, 255, 20)

with col3:
    c_scale = st.slider("Contraste LASCA", 1, 100, 30)

with col4:
    k_size = st.slider("Tamanho do Kernel", 3, 15, 5, step=2)

# --- PROCESSAMENTO ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if modo == "CINZA":
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        img_f32 = gray.astype(np.float32)
        kernel = np.ones((k_size, k_size), np.float32) / (k_size**2)
        mean = cv2.filter2D(img_f32, -1, kernel)
        sq_mean = cv2.filter2D(img_f32**2, -1, kernel)
        std = cv2.sqrt(cv2.absdiff(sq_mean, mean**2))
        mean[mean == 0] = 1
        lasca = (std / mean) * (255.0 / (c_scale / 50.0))
        lasca_u8 = 255 - np.clip(lasca, 0, 255).astype(np.uint8)
        lasca_u8[mean < m_gray] = 0
        result = cv2.applyColorMap(lasca_u8, cv2.COLORMAP_JET)
    
    return frame.from_ndarray(result, format="bgr24")

# --- ÁREA DO VÍDEO COM ESCALA LATERAL ---
# Criamos um layout HTML/CSS para colocar a escala ao lado do vídeo
st.markdown('<div class="video-flex-container">', unsafe_allow_html=True)

# O vídeo propriamente dito
webrtc_streamer(
    key="biospeckle-main",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {"facingMode": escolha_camera},
        "audio": False
    },
    async_processing=True,
)

# A Escala de Cores (HTML)
st.markdown("""
    <div class="color-bar">
        <span>MAX</span>
        <span>|</span>
        <span>|</span>
        <span>|</span>
        <span>MIN</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

st.write("---")
st.caption("BiospecklePy Web - Escala de Intensidade de Fluxo Ativa")