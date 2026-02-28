import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np

st.set_page_config(page_title="BiospecklePy", layout="wide")

# Estilo Verde para Sliders
st.markdown("""
    <style>
    .stSlider [data-baseweb="slider"] [role="slider"] { background-color: #2D5A27; }
    .stSlider [data-baseweb="slider"] [aria-valuemax] { background-color: #2D5A27; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 BiospecklePy")

# --- CONTROLES SOLICITADOS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    modo = st.radio("Modo de Visão:", ["LASCA", "CINZA"])
    min_cinza = st.slider("MÍN_CINZA", 0, 255, 20)

with col2:
    contraste = st.slider("CONTRASTE", 1, 100, 30)
    # Exposição: Geralmente valores negativos em navegadores representam compensação
    exposicao = st.slider("EXPOSIÇÃO", -10, 10, 0)

with col3:
    # Ganho (ISO/Sensibilidade)
    ganho = st.slider("GANHO", 0, 100, 0)
    kernel_size = st.slider("KERNEL (Suavização)", 3, 15, 5, step=2)

with col4:
    escolha_camera = st.selectbox("Câmera:", ["user", "environment"], 
                                 format_func=lambda x: "Frontal" if x == "user" else "Externa")

# --- FUNÇÃO DA ESCALA DE CORES INTERNA ---
def desenhar_escala(img):
    h, w = img.shape[:2]
    bar_w, bar_h = int(w * 0.05), int(h * 0.6)
    x_offset, y_offset = w - bar_w - 20, int((h - bar_h) / 2)

    escala_cinza = np.linspace(0, 255, bar_h).astype(np.uint8).reshape(-1, 1)
    escala_cinza = np.repeat(escala_cinza, bar_w, axis=1)
    escala_cinza = cv2.flip(escala_cinza, 0)
    barra_colorida = cv2.applyColorMap(escala_cinza, cv2.COLORMAP_JET)

    cv2.rectangle(img, (x_offset-2, y_offset-2), (x_offset+bar_w+2, y_offset+bar_h+2), (255,255,255), 1)
    img[y_offset:y_offset+bar_h, x_offset:x_offset+bar_w] = barra_colorida
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "MAX", (x_offset - 35, y_offset + 10), font, 0.4, (255,255,255), 1)
    cv2.putText(img, "MIN", (x_offset - 30, y_offset + bar_h), font, 0.4, (255,255,255), 1)
    return img

# --- PROCESSAMENTO ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if modo == "CINZA":
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        img_f32 = gray.astype(np.float32)
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        mean = cv2.filter2D(img_f32, -1, kernel)
        sq_mean = cv2.filter2D(img_f32**2, -1, kernel)
        std = cv2.sqrt(cv2.absdiff(sq_mean, mean**2))
        
        mean[mean == 0] = 1
        lasca = (std / mean) * (255.0 / (contraste / 50.0))
        lasca_u8 = 255 - np.clip(lasca, 0, 255).astype(np.uint8)
        lasca_u8[mean < min_cinza] = 0
        
        result = cv2.applyColorMap(lasca_u8, cv2.COLORMAP_JET)
        result = desenhar_escala(result)
    
    return frame.from_ndarray(result, format="bgr24")

# --- VÍDEO COM RESTRIÇÕES DE HARDWARE ---
webrtc_streamer(
    key="biospeckle",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "facingMode": escolha_camera,
            # Tentativa de aplicar Exposição e Ganho via Constraints (Pode não funcionar em todas as câmeras)
            "exposureMode": "manual",
            "exposureCompensation": exposicao,
            "iso": ganho
        },
        "audio": False
    },
    async_processing=True,
)

st.caption("BiospecklePy Web - Controles de Hardware e Algoritmo Ativos")