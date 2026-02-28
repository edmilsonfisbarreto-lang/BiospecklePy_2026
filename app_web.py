import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Configuração da Página
st.set_page_config(page_title="BiospecklePy - LASCA", layout="wide")

def jet_colormap(value):
    """Simula o mapeamento Jet (0-1) para BGR"""
    return cv2.applyColorMap((value * 255).astype(np.uint8), cv2.COLORMAP_JET)

class BiospeckleTransformer(VideoTransformerBase):
    def __init__(self):
        self.mode = "FLUXO"
        self.min_gray = 10
        self.contrast_scale = 20
        self.gain = 1.0
        self.exposure = 1.0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Ajuste de Ganho e "Exposição" (Brilho/Contraste linear)
        img = cv2.convertScaleAbs(img, alpha=self.gain, beta=self.exposure * 10)
        
        # 2. Conversão para Escala de Cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.mode == "CINZA":
            # Retorna imagem em tons de cinza (convertida de volta para BGR para o streamer)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 3. Lógica LASCA (Modo FLUXO)
        # Usamos uma janela de 5x5 para calcular desvio padrão e média
        gray_f = gray.astype(np.float32)
        
        # Média Local (mu)
        mu = cv2.blur(gray_f, (5, 5))
        
        # Desvio Padrão Local (sigma)
        # sigma = sqrt( E[X^2] - (E[X])^2 )
        mu_sq = cv2.blur(gray_f**2, (5, 5))
        sigma = np.sqrt(np.maximum(0, mu_sq - mu**2))
        
        # Cálculo do Contraste (K = sigma / mu)
        # Evita divisão por zero e aplica o threshold de cinza mínimo
        with np.errstate(divide='ignore', invalid='ignore'):
            k = sigma / mu
            k[mu < self.min_gray] = 0
            k = np.nan_to_num(k)

        # Normalização baseada no slider de contraste (0.01 a 1.0)
        max_c = max(0.01, self.contrast_scale / 100.0)
        lasca = np.clip(k / max_c, 0, 1)
        
        # Inverte (1 - lasca) para seguir sua lógica original e aplica ColorMap
        lasca_inverted = (1.0 - lasca)
        color_mapped = cv2.applyColorMap((lasca_inverted * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return color_mapped

# --- INTERFACE STREAMLIT ---
st.sidebar.title("🧪 BiospecklePy")
st.sidebar.caption("Sistema de Análise LASCA")

mode = st.sidebar.radio("VISUALIZAÇÃO", ["FLUXO", "CINZA"])
min_gray = st.sidebar.slider("MÍN CINZA", 0, 255, 10)
contrast = st.sidebar.slider("CONTRASTE", 1, 100, 20)
gain = st.sidebar.slider("GANHO", 1.0, 5.0, 1.0, 0.1)
exposure = st.sidebar.slider("EXPOSIÇÃO", 1.0, 10.0, 5.0, 0.5)

# Inicialização do Stream de Vídeo
ctx = webrtc_streamer(
    key="biospeckle",
    video_transformer_factory=BiospeckleTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# Atualiza os parâmetros do processador em tempo real
if ctx.video_transformer:
    ctx.video_transformer.mode = mode
    ctx.video_transformer.min_gray = min_gray
    ctx.video_transformer.contrast_scale = contrast
    ctx.video_transformer.gain = gain
    ctx.video_transformer.exposure = exposure

st.sidebar.markdown("---")
st.sidebar.write("Desenvolvido por Edmilson Souza Barreto")