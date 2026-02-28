import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.set_page_config(page_title="BiospecklePy", layout="wide")

class BiospeckleTransformer(VideoTransformerBase):
    def __init__(self):
        self.mode = "FLUXO"
        self.min_gray = 10
        self.contrast_scale = 20
        self.gain = 0
        self.exposure = 5

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # --- LÓGICA ORIGINAL DE GANHO E EXPOSIÇÃO ---
        gain_factor = 1 + self.gain / 740
        exposure_factor = self.exposure / 5
        total_factor = gain_factor * exposure_factor
        
        # Conversão para Cinza (Pesos originais: 0.299R, 0.587G, 0.114B)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        if self.mode == "CINZA":
            # Aplica os fatores e limita a 255
            res_gray = np.clip(gray * total_factor, 0, 255).astype(np.uint8)
            return cv2.cvtColor(res_gray, cv2.COLOR_GRAY2BGR)

        # --- MODO FLUXO (LASCA) ---
        # Aplica ganho/exposição antes do cálculo
        gray_f = gray * total_factor
        
        # Kernel 5x5 como no React original
        kernel_size = 5
        mu = cv2.blur(gray_f, (kernel_size, kernel_size))
        mu_sq = cv2.blur(gray_f**2, (kernel_size, kernel_size))
        variance = mu_sq - mu**2
        sigma = np.sqrt(np.maximum(0, variance))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # maxC no seu código era contrast / 100
            max_c = max(0.01, self.contrast_scale / 100.0)
            # lasca = (std / mean) / maxC
            k = np.divide(sigma, mu, out=np.zeros_like(sigma), where=mu > 0)
            lasca = k / max_c
            
            # Filtro de cinza mínimo (mean < currentMinGray)
            lasca[mu < self.min_gray] = 1.0 # 1.0 resultará em preto após o (1-lasca)
            lasca = np.clip(lasca, 0, 1)

        # Mapeamento JET (1 - lasca)
        # 0 = Azul (Baixo), 1 = Vermelho (Alto)
        color_mapped = cv2.applyColorMap(((1.0 - lasca) * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # --- BARRA DE CORES LATERAL ---
        bar_w, bar_h = 20, 200
        start_y, start_x = (h - bar_h) // 2, w - bar_w - 20
        
        gradient = np.linspace(0, 255, bar_h).astype(np.uint8).reshape(bar_h, 1)
        # Invertemos o gradiente para bater com a lógica do vídeo (Alto em cima)
        gradient_color = cv2.applyColorMap(cv2.flip(gradient, 0), cv2.COLORMAP_JET)
        
        cv2.rectangle(color_mapped, (start_x-2, start_y-2), (start_x+bar_w+2, start_y+bar_h+2), (255, 255, 255), 1)
        color_mapped[start_y:start_y+bar_h, start_x:start_x+bar_w] = gradient_color
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(color_mapped, "Alto", (start_x - 45, start_y + 15), font, 0.5, (255, 255, 255), 1)
        cv2.putText(color_mapped, "Baixo", (start_x - 50, start_y + bar_h), font, 0.5, (255, 255, 255), 1)

        return color_mapped

# --- INTERFACE ---
st.markdown("<h3 style='text-align: center;'>BiospecklePy - Sistema de Análise LASCA</h3>", unsafe_allow_html=True)

# Controles Superiores
c1, c2, c3 = st.columns([2, 4, 4])

with c1:
    cam_id = st.selectbox("CÂMERA", [0, 1, 2], format_func=lambda x: f"Dispositivo {x}")
    mode = st.radio("VISUALIZAÇÃO", ["FLUXO", "CINZA"], horizontal=True)

with c2:
    gain = st.slider("GANHO", 0, 740, 0)
    exposure = st.slider("EXPOSIÇÃO", 1, 13, 5)

with c3:
    contrast = st.slider("CONTRASTE", 1, 100, 20)
    min_gray = st.slider("MÍN CINZA", 0, 255, 10)

st.divider()

# Vídeo
v_col1, v_col2, v_col3 = st.columns([1, 10, 1])
with v_col2:
    ctx = webrtc_streamer(
        key=f"biospeckle-{cam_id}",
        video_transformer_factory=BiospeckleTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"deviceId": str(cam_id)} if cam_id > 0 else True, "audio": False},
        async_processing=True,
    )

    if ctx.video_transformer:
        ctx.video_transformer.mode = mode
        ctx.video_transformer.min_gray = min_gray
        ctx.video_transformer.contrast_scale = contrast
        ctx.video_transformer.gain = gain
        ctx.video_transformer.exposure = exposure

st.caption("Desenvolvido por Edmilson Souza Barreto")