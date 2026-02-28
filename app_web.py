import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Configuração da página para usar a largura total
st.set_page_config(page_title="BiospecklePy", layout="wide")

class BiospeckleTransformer(VideoTransformerBase):
    def __init__(self):
        self.mode = "FLUXO"
        self.min_gray = 10
        self.contrast_scale = 20
        self.gain = 1.0
        self.exposure = 1.0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Ajuste de Ganho e Brilho (Exposição simulada)
        img = cv2.convertScaleAbs(img, alpha=self.gain, beta=self.exposure * 10)
        
        # Conversão para Cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.mode == "CINZA":
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Lógica LASCA (Cálculo de Bio-speckle)
        gray_f = gray.astype(np.float32)
        mu = cv2.blur(gray_f, (5, 5))
        mu_sq = cv2.blur(gray_f**2, (5, 5))
        sigma = np.sqrt(np.maximum(0, mu_sq - mu**2))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            k = sigma / mu
            k[mu < self.min_gray] = 0
            k = np.nan_to_num(k)

        max_c = max(0.01, self.contrast_scale / 100.0)
        lasca = np.clip(k / max_c, 0, 1)
        
        # Mapeamento de Cores JET (Invertido para destacar atividade em vermelho)
        color_mapped = cv2.applyColorMap(((1.0 - lasca) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return color_mapped

# --- INTERFACE SUPERIOR ---
st.title("🧪 BiospecklePy - Análise LASCA")

# Linha 1: Seleção de Modo e Controles Principais
col1, col2, col3, col4 = st.columns([2, 2, 3, 3])

with col1:
    mode = st.radio("MODO", ["FLUXO", "CINZA"], horizontal=True)

with col2:
    # Nota: No Streamlit Cloud, a seleção de câmera é feita pelo ícone de engrenagem 
    # dentro do componente WebRTC, mas definimos o componente abaixo.
    st.write("**CÂMERA ATIVA**")
    st.caption("Use o menu do vídeo abaixo para trocar de câmera.")

with col3:
    gain = st.slider("GANHO", 1.0, 5.0, 1.0, 0.1)
    exposure = st.slider("EXPOSIÇÃO", 1.0, 10.0, 5.0, 0.5)

with col4:
    contrast = st.slider("CONTRASTE (LASCA)", 1, 100, 20)
    min_gray = st.slider("THRESHOLD CINZA", 0, 255, 10)

st.divider()

# --- ÁREA DO VÍDEO ---
# Centralizando o vídeo
v_col1, v_col2, v_col3 = st.columns([1, 10, 1])

with v_col2:
    ctx = webrtc_streamer(
        key="biospeckle-system",
        video_transformer_factory=BiospeckleTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": True, # Isso habilita a seleção de câmera no cliente
            "audio": False
        },
        async_processing=True,
    )

    # Passando os valores dos Sliders para o processador de imagem
    if ctx.video_transformer:
        ctx.video_transformer.mode = mode
        ctx.video_transformer.min_gray = min_gray
        ctx.video_transformer.contrast_scale = contrast
        ctx.video_transformer.gain = gain
        ctx.video_transformer.exposure = exposure

# Rodapé
st.markdown(
    """
    <style>
    .footer { position: fixed; bottom: 0; width: 100%; text-align: center; color: gray; font-size: 12px; }
    </style>
    <div class="footer">Desenvolvido por Edmilson Souza Barreto | BiospecklePy 2026</div>
    """, unsafe_allow_html=True
)