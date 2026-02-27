import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np

# Configuração da página
st.set_page_config(page_title="BiospecklePy Web", layout="wide")

st.title("🌱 BiospecklePy - Real Time")

# --- BARRA LATERAL (SLIDERS) ---
st.sidebar.header("Configurações do Algoritmo")

# Usamos chaves (keys) para garantir que o Streamlit mantenha os valores
m_gray = st.sidebar.slider("Filtro de Ruído (Brilho)", 0, 255, 20, key="slider_gray")
c_scale = st.sidebar.slider("Contraste LASCA", 1, 100, 30, key="slider_contrast")
k_size = st.sidebar.slider("Tamanho do Kernel (Suavização)", 3, 15, 5, step=2, key="slider_kernel")

# Configuração de rede para evitar travamentos
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def video_frame_callback(frame):
    # 1. Converte o frame da câmera para array (BGR)
    img = frame.to_ndarray(format="bgr24")
    
    # 2. Processamento Cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_f32 = gray.astype(np.float32)

    # 3. Lógica LASCA
    # Criamos o kernel baseado no slider lateral
    kernel = np.ones((k_size, k_size), np.float32) / (k_size**2)
    
    mean = cv2.filter2D(img_f32, -1, kernel)
    sq_mean = cv2.filter2D(img_f32**2, -1, kernel)
    
    # Variância e Desvio Padrão
    std = cv2.sqrt(cv2.absdiff(sq_mean, mean**2))
    
    # Evita divisão por zero
    mean[mean == 0] = 1
    
    # Cálculo do Contraste ajustado pelo slider
    lasca = (std / mean) * (255.0 / (c_scale / 50.0))
    
    # 4. Finalização e Colorização
    lasca_u8 = 255 - np.clip(lasca, 0, 255).astype(np.uint8)
    
    # Aplica o filtro de brilho mínimo do slider
    lasca_u8[mean < m_gray] = 0
    
    # Aplica Mapa de Cores JET (Igual ao seu EXE)
    result = cv2.applyColorMap(lasca_u8, cv2.COLORMAP_JET)
    
    return frame.from_ndarray(result, format="bgr24")

# --- EXIBIÇÃO DO VÍDEO ---
webrtc_streamer(
    key="biospeckle-analysis",
    mode=WebRtcMode.SENDRECV, # Envia e recebe vídeo
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True,
        "audio": False  # <--- AQUI DESATIVAMOS O ÁUDIO
    },
    async_processing=True, # Melhora a performance dos sliders
)

st.info("Ajuste os sliders na barra lateral à esquerda para modificar o mapa de fluxo em tempo real.")