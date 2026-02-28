import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np

st.set_page_config(page_title="BiospecklePy", layout="wide")

# CSS para Sliders Verdes
st.markdown("""
    <style>
    .stSlider [data-baseweb="slider"] [role="slider"] { background-color: #2D5A27; }
    .stSlider [data-baseweb="slider"] [aria-valuemax] { background-color: #2D5A27; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 BiospecklePy")

# --- CONTROLES SUPERIORES ---
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    modo = st.radio("Modo de Visão:", ["FLUXO", "CINZA"])
    escolha_camera = st.selectbox("Câmera:", ["user", "environment"], 
                                   format_func=lambda x: "Frontal" if x == "user" else "Externa/Traseira")

with col2:
    gain = st.slider("GANHO", 0, 740, 0)
    exposure = st.slider("EXPOSIÇÃO", 1, 13, 5)

with col3:
    m_gray = st.slider("Mín Cinza", 0, 255, 10)

with col4:
    c_scale = st.slider("Contraste LASCA", 1, 100, 20)

with col5:
    k_size = st.slider("Tamanho do Kernel", 3, 15, 5, step=2)

# --- FUNÇÃO DA ESCALA LATERAL (INVERTIDA) ---
def desenhar_escala(img):
    h, w = img.shape[:2]
    bar_w = int(w * 0.03)
    bar_h = int(h * 0.5)
    x_offset = w - bar_w - 30
    y_offset = int((h - bar_h) / 2)

    # Gradiente de 0 a 255
    escala_cinza = np.linspace(0, 255, bar_h).astype(np.uint8).reshape(-1, 1)
    escala_cinza = np.repeat(escala_cinza, bar_w, axis=1)
    
    # Mapeamento JET (Nesta configuração: 255=Azul no topo, 0=Vermelho na base)
    barra_colorida = cv2.applyColorMap(escala_cinza, cv2.COLORMAP_JET)
    
    cv2.rectangle(img, (x_offset-2, y_offset-2), (x_offset+bar_w+2, y_offset+bar_h+2), (255,255,255), 1)
    img[y_offset:y_offset+bar_h, x_offset:x_offset+bar_w] = barra_colorida
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "ALTO (AZUL)", (x_offset - 85, y_offset + 10), font, 0.4, (255,255,255), 1)
    cv2.putText(img, "BAIXO (VERM)", (x_offset - 90, y_offset + bar_h), font, 0.4, (255,255,255), 1)
    return img

# --- PROCESSAMENTO DE VÍDEO ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Ajustes de Brilho Originais
    total_f = (1 + gain / 740) * (exposure / 5)
    gray_f32 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) * total_f
    
    if modo == "CINZA":
        result_u8 = np.clip(gray_f32, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result_u8, cv2.COLOR_GRAY2BGR)
    else:
        # Lógica LASCA
        kernel = np.ones((k_size, k_size), np.float32) / (k_size**2)
        mean = cv2.filter2D(gray_f32, -1, kernel)
        sq_mean = cv2.filter2D(gray_f32**2, -1, kernel)
        std = cv2.sqrt(cv2.max(0.0, sq_mean - mean**2))
        
        max_c = max(0.01, c_scale / 100.0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # K = std / mean
            k = np.divide(std, mean, out=np.zeros_like(std), where=mean > 0)
            lasca = k / max_c
            
            # Filtro de ruído: áreas escuras ficam com valor que resulta em Vermelho (Baixo)
            lasca[mean < m_gray] = 0.0 
            
            # Normalização (0.0 a 1.0)
            # Como você quer ALTO = AZUL, e no JET o azul está perto de 255:
            # Vamos mapear lasca diretamente para o colormap sem inverter (1-lasca)
            lasca_u8 = (np.clip(lasca, 0, 1) * 255).astype(np.uint8)
            
        result = cv2.applyColorMap(lasca_u8, cv2.COLORMAP_JET)
        result = desenhar_escala(result)

    return frame.from_ndarray(result, format="bgr24")

# --- EXECUÇÃO DO STREAMER ---
# A key dinâmica garante que o componente reinicie ao trocar a câmera no selectbox
webrtc_streamer(
    key=f"biospeckle-stream-{escolha_camera}",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "facingMode": {"exact": escolha_camera}
        }, 
        "audio": False
    },
    async_processing=True,
)

st.caption("BiospecklePy Web - Escala: Azul (Alta Atividade) | Vermelho (Baixa Atividade)")