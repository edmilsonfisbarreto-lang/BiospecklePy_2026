import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np

# Configuração da página
st.set_page_config(page_title="BiospecklePy", layout="wide")

# CSS Estilo Verde (Sliders e Botões)
st.markdown("""
    <style>
    .stSlider [data-baseweb="slider"] [role="slider"] { background-color: #2D5A27; }
    .stSlider [data-baseweb="slider"] [aria-valuemax] { background-color: #2D5A27; }
    .stButton>button { 
        background-color: #2D5A27; 
        color: white; 
        border-radius: 8px; 
        width: 100%; 
        height: 3.5em; 
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover { background-color: #3d7a35; color: white; }
    #video-container { position: relative; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 BiospecklePy")

# --- CONFIGURAÇÃO DE REDE ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- CONTROLES (UNIFICADOS) ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    modo = st.radio("Modo de Visão:", ["LASCA", "CINZA"])
    # Seleção única de câmera
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

# --- PLAYER DE VÍDEO ---
# O vídeo agora obedece cegamente à seleção do Selectbox acima
webrtc_streamer(
    key="biospeckle-main",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {"facingMode": {"exact": escolha_camera} if escolha_camera == "environment" else escolha_camera},
        "audio": False
    },
    async_processing=True,
)

st.write("---")

# --- BOTÃO DE CAPTURA (MÉTODO CANVAS INVISÍVEL) ---
if st.button("📸 CAPTURAR E SALVAR IMAGEM"):
    st.components.v1.html("""
        <script>
        // Função para encontrar o vídeo mesmo dentro de componentes complexos
        const videos = window.parent.document.querySelectorAll("video");
        const targetVideo = videos[0]; 

        if (targetVideo && targetVideo.readyState === 4) {
            const canvas = document.createElement("canvas");
            canvas.width = targetVideo.videoWidth;
            canvas.height = targetVideo.videoHeight;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(targetVideo, 0, 0, canvas.width, canvas.height);
            
            const dataURL = canvas.toDataURL("image/png");
            const link = document.createElement("a");
            link.href = dataURL;
            link.download = "biospeckle_web_capture.png";
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } else {
            alert("Erro: Câmera não detectada ou ainda carregando. Clique em START primeiro.");
        }
        </script>
        """, height=0)

st.caption("BiospecklePy Web - Versão Integrada")