import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np
from PIL import Image
import io

# Configuração da página
st.set_page_config(page_title="BiospecklePy", layout="wide")

# CSS para deixar sliders e botões verdes (Corrigido para unsafe_allow_html)
st.markdown("""
    <style>
    /* Cor dos Sliders */
    .stSlider [data-baseweb="slider"] [role="slider"] { background-color: #2D5A27; }
    .stSlider [data-baseweb="slider"] [aria-valuemax] { background-color: #2D5A27; }
    div[data-testid="stThumbValue"] { color: #2D5A27; }
    
    /* Cor dos Botões */
    div.stButton > button:first-child {
        background-color: #2D5A27;
        color: white;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #3d7a35;
        border: none;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 BiospecklePy")

# --- CONFIGURAÇÃO DE REDE ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Inicializa o estado da sessão
if "frame_capturado" not in st.session_state:
    st.session_state["frame_capturado"] = None

# --- INTERFACE DE CONTROLE EM COLUNAS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    modo = st.radio("Modo de Visão:", ["LASCA", "CINZA"], key="modo_visao")
    camera_facing = st.selectbox("Câmera:", ["user", "environment"], 
                                 format_func=lambda x: "Frontal/Padrão" if x == "user" else "Traseira/Externa")

with col2:
    m_gray = st.slider("Filtro de Ruído", 0, 255, 20)

with col3:
    c_scale = st.slider("Contraste LASCA", 1, 100, 30)

with col4:
    k_size = st.slider("Tamanho do Kernel", 3, 15, 5, step=2)

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
    
    # Armazena o frame atual para captura
    st.session_state["frame_capturado"] = result
    
    return frame.from_ndarray(result, format="bgr24")

# --- COMPONENTE DE VÍDEO ---
webrtc_streamer(
    key="biospeckle",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": {"facingMode": camera_facing}, "audio": False},
    async_processing=True,
)

st.write("---")

# --- SISTEMA DE CAPTURA ---
# O botão verde agora processa o frame salvo na sessão
if st.button("📸 PREPARAR CAPTURA DA IMAGEM"):
    if st.session_state["frame_capturado"] is not None:
        rgb_img = cv2.cvtColor(st.session_state["frame_capturado"], cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_img)
        buf = io.BytesIO()
        img_pil.save(buf, format="TIFF")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="💾 CLIQUE AQUI PARA BAIXAR (.TIF)",
            data=byte_im,
            file_name="captura_biospeckle.tif",
            mime="image/tiff"
        )
    else:
        st.error("Aguarde a câmera carregar e o vídeo aparecer antes de capturar.")

st.caption("BiospecklePy Web v2.1")