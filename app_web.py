import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np
from PIL import Image
import io

# Configuração da página e Estilo Verde (CSS Customizado)
st.set_page_config(page_title="BiospecklePy", layout="wide")

# Aplica CSS para deixar sliders e botões verdes
st.markdown("""
    <style>
    .stSlider [data-baseweb="slider"] [role="slider"] { background-color: #2D5A27; }
    .stSlider [data-baseweb="slider"] [aria-valuemax] { background-color: #2D5A27; }
    div.stButton > button:first-child {
        background-color: #2D5A27;
        color: white;
        border-radius: 5px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #3d7a35;
        color: white;
    }
    </style>
    """, unsafe_allow_dict=True)

st.title("🌱 BiospecklePy")

# --- CONFIGURAÇÃO DE REDE ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Inicializa o buffer de captura no estado da sessão
if "frame_capturado" not in st.session_state:
    st.session_state["frame_capturado"] = None

# --- INTERFACE DE CONTROLE ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    modo = st.radio("Modo de Visão:", ["LASCA", "CINZA"], key="modo_visao")
    camera_facing = st.selectbox("Câmera:", ["user", "environment"], 
                                 format_func=lambda x: "Frontal/Padrão" if x == "user" else "Traseira/Externa")

with col2:
    m_gray = st.slider("Filtro de Ruído", 0, 255, 20)
    c_scale = st.slider("Contraste LASCA", 1, 100, 30)

with col3:
    k_size = st.slider("Tamanho do Kernel", 3, 15, 5, step=2)

# --- FUNÇÃO DE PROCESSAMENTO ---
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
    
    # SALVAMENTO PERSISTENTE
    st.session_state["frame_capturado"] = result
    
    return frame.from_ndarray(result, format="bgr24")

# --- VÍDEO ---
webrtc_streamer(
    key="biospeckle",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": {"facingMode": camera_facing}, "audio": False},
    async_processing=True,
)

st.write("---")

# --- SISTEMA DE CAPTURA CORRIGIDO ---
if st.button("📸 GERAR ARQUIVO DE CAPTURA"):
    if st.session_state["frame_capturado"] is not None:
        # Processa a imagem para download
        rgb_img = cv2.cvtColor(st.session_state["frame_capturado"], cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_img)
        buf = io.BytesIO()
        img_pil.save(buf, format="TIFF")
        
        st.download_button(
            label="💾 BAIXAR AGORA (.TIF)",
            data=buf.getvalue(),
            file_name="captura_biospeckle.tif",
            mime="image/tiff"
        )
    else:
        st.error("O sistema ainda não recebeu quadros da câmera. Inicie o vídeo e aguarde 2 segundos.")

st.caption("BiospecklePy Web v2.0")