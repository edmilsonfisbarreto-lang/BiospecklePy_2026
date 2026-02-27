import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np
from PIL import Image
import io
import threading

# Configuração da página
st.set_page_config(page_title="BiospecklePy", layout="wide")

# CSS para Estilo Verde
st.markdown("""
    <style>
    .stSlider [data-baseweb="slider"] [role="slider"] { background-color: #2D5A27; }
    .stSlider [data-baseweb="slider"] [aria-valuemax] { background-color: #2D5A27; }
    div.stButton > button:first-child {
        background-color: #2D5A27;
        color: white;
        border-radius: 8px;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 BiospecklePy")

# --- LÓGICA DE CAPTURA PERSISTENTE ---
# Usamos um lock para evitar conflitos de leitura/escrita de memória
lock = threading.Lock()
img_container = {"img": None}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Processamento LASCA/CINZA
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
    
    # Salva o frame de forma segura
    with lock:
        img_container["img"] = result
        
    return frame.from_ndarray(result, format="bgr24")

# --- INTERFACE ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    modo = st.radio("Modo:", ["LASCA", "CINZA"])
    camera_facing = st.selectbox("Câmera:", ["user", "environment"], format_func=lambda x: "Frontal" if x=="user" else "Externa")
with col2:
    m_gray = st.slider("Filtro Ruído", 0, 255, 20)
with col3:
    c_scale = st.slider("Contraste", 1, 100, 30)
with col4:
    k_size = st.slider("Kernel", 3, 15, 5, step=2)

# --- VÍDEO ---
webrtc_streamer(
    key="biospeckle",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": {"facingMode": camera_facing}, "audio": False},
    async_processing=True,
)

st.write("---")

# --- CAPTURA CORRIGIDA ---
if st.button("📸 PREPARAR CAPTURA"):
    with lock:
        frame_para_download = img_container["img"]
    
    if frame_para_download is not None:
        rgb_img = cv2.cvtColor(frame_para_download, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_img)
        buf = io.BytesIO()
        img_pil.save(buf, format="TIFF")
        
        st.download_button(
            label="💾 BAIXAR IMAGEM (.TIF)",
            data=buf.getvalue(),
            file_name="biospeckle_web.tif",
            mime="image/tiff"
        )
    else:
        st.warning("Aguarde o vídeo carregar. Se o erro persistir, clique em START e espere 3 segundos.")

st.caption("BiospecklePy v2.2 - Estável")