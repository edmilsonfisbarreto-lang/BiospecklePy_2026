import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np

# Configuração da página
st.set_page_config(page_title="BiospecklePy", layout="wide")

# Estilo Visual Verde
st.markdown("""
    <style>
    .stSlider [data-baseweb="slider"] [role="slider"] { background-color: #2D5A27; }
    .stSlider [data-baseweb="slider"] [aria-valuemax] { background-color: #2D5A27; }
    .stButton>button { background-color: #2D5A27; color: white; border-radius: 8px; width: 100%; height: 3em; font-weight: bold; }
    .stButton>button:hover { background-color: #3d7a35; color: white; border: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 BiospecklePy")

# --- INTERFACE DE CONTROLE EM COLUNAS ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    modo = st.radio("Modo de Visão:", ["LASCA", "CINZA"])
    camera_facing = st.selectbox("Câmera:", ["user", "environment"], format_func=lambda x: "Frontal" if x=="user" else "Externa")
with col2:
    m_gray = st.slider("Filtro de Ruído", 0, 255, 20)
with col3:
    c_scale = st.slider("Contraste LASCA", 1, 100, 30)
with col4:
    k_size = st.slider("Tamanho do Kernel", 3, 15, 5, step=2)

# --- LÓGICA DE PROCESSAMENTO ---
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

# --- EXIBIÇÃO DO VÍDEO ---
webrtc_streamer(
    key="biospeckle",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": {"facingMode": camera_facing}, "audio": False},
    async_processing=True,
)

# --- BOTÃO DE CAPTURA VIA JAVASCRIPT (CLIENT-SIDE) ---
st.write("---")
st.markdown("""
    <script>
    function captureImage() {
        const video = document.querySelector('video');
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        const link = document.createElement('a');
        link.download = 'biospeckle_capture.png';
        link.href = canvas.toDataURL('image/png');
        link.click();
    }
    </script>
    """, unsafe_allow_html=True)

if st.button("📸 CAPTURAR IMAGEM AGORA"):
    st.components.v1.html("""
        <script>
        const video = window.parent.document.querySelector('video');
        if (video) {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const link = document.createElement('a');
            link.download = 'biospeckle_capture.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        } else {
            alert("Inicie a câmera antes de capturar!");
        }
        </script>
        """, height=0)

st.caption("BiospecklePy Web - Captura Instantânea via Navegador")