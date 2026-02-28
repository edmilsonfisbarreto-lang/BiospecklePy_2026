import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Configuração da página
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
        h, w, _ = img.shape
        
        # Ajuste de Ganho e Brilho
        img = cv2.convertScaleAbs(img, alpha=self.gain, beta=self.exposure * 10)
        
        # Conversão para Cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.mode == "CINZA":
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Lógica LASCA
        gray_f = gray.astype(np.float32)
        mu = cv2.blur(gray_f, (7, 7)) # Kernel 7x7 para maior estabilidade
        mu_sq = cv2.blur(gray_f**2, (7, 7))
        sigma = np.sqrt(np.maximum(0, mu_sq - mu**2))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            k = sigma / mu
            k[mu < self.min_gray] = 0
            k = np.nan_to_num(k)

        max_c = max(0.01, self.contrast_scale / 100.0)
        lasca = np.clip(k / max_c, 0, 1)
        
        # Mapeamento de Cores JET
        processed = cv2.applyColorMap(((1.0 - lasca) * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # --- ADIÇÃO DA BARRA DE CORES (COLORBAR) ---
        bar_w = 20
        bar_h = 200
        # Posição centralizada à direita
        start_y = (h - bar_h) // 2
        start_x = w - bar_w - 20
        
        # Criar gradiente da barra
        gradient = np.linspace(0, 255, bar_h).astype(np.uint8).reshape(bar_h, 1)
        gradient = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        
        # Desenhar borda branca para a barra
        cv2.rectangle(processed, (start_x-2, start_y-2), (start_x+bar_w+2, start_y+bar_h+2), (255, 255, 255), 1)
        # Aplicar a barra no frame
        processed[start_y:start_y+bar_h, start_x:start_x+bar_w] = gradient
        
        # Textos da Barra
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(processed, "ALTO", (start_x - 45, start_y + 15), font, 0.5, (255, 255, 255), 1)
        cv2.putText(processed, "BAIXO", (start_x - 50, start_y + bar_h), font, 0.5, (255, 255, 255), 1)

        return processed

# --- INTERFACE SUPERIOR ---
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>🧪 BiospecklePy - LASCA Analysis</h2>", unsafe_allow_html=True)

# Painel de Controle Superior
with st.container():
    c1, c2, c3 = st.columns([2, 4, 4])
    
    with c1:
        st.write("**CÂMERA E MODO**")
        # Simulação de seleção de câmera (ID do dispositivo)
        # Nota: No Streamlit Cloud, o browser pedirá permissão para a lista de dispositivos
        cam_id = st.selectbox("Selecione o Dispositivo", [0, 1, 2], format_func=lambda x: f"Câmera {x}")
        mode = st.radio("Visualização", ["FLUXO", "CINZA"], horizontal=True)

    with c2:
        st.write("**AJUSTES DE IMAGEM**")
        gain = st.slider("GANHO", 1.0, 5.0, 1.0, 0.1)
        exposure = st.slider("EXPOSIÇÃO", 1.0, 10.0, 5.0, 0.5)

    with c3:
        st.write("**PARÂMETROS LASCA**")
        contrast = st.slider("CONTRASTE (K)", 1, 100, 20)
        min_gray = st.slider("THRESHOLD CINZA", 0, 255, 10)

st.divider()

# --- ÁREA DO VÍDEO ---
# Centralizando o vídeo na tela
v_col1, v_col2, v_col3 = st.columns([1, 10, 1])

with v_col2:
    ctx = webrtc_streamer(
        key=f"biospeckle-cam-{cam_id}", # A chave muda conforme a câmera para forçar o restart
        video_transformer_factory=BiospeckleTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {"deviceId": str(cam_id)} if cam_id > 0 else True,
            "audio": False
        },
        async_processing=True,
    )

    if ctx.video_transformer:
        ctx.video_transformer.mode = mode
        ctx.video_transformer.min_gray = min_gray
        ctx.video_transformer.contrast_scale = contrast
        ctx.video_transformer.gain = gain
        ctx.video_transformer.exposure = exposure

# Rodapé simples
st.caption(f"Status: {'🟢 Ativo' if ctx.state.playing else '🔴 Offline'} | Desenvolvido por Edmilson Souza Barreto")