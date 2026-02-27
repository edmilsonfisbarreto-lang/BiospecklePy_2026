import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np

st.set_page_config(page_title="BiospecklePy Live", layout="wide")

st.title("🌱 BiospecklePy - Análise em Tempo Real")

# Barra Lateral com os controles (igual ao seu .exe)
st.sidebar.header("Parâmetros de Análise")
min_gray = st.sidebar.slider("Mínimo Cinza", 0, 255, 10)
contrast_scale = st.sidebar.slider("Contraste", 1, 100, 20)
kernel_size = st.sidebar.slider("Tamanho do Kernel", 3, 15, 5, step=2)

class BiospeckleTransformer(VideoTransformerBase):
    def __init__(self):
        self.kernel_size = kernel_size

    def transform(self, frame):
        # Converte o frame da câmera para array numpy
        img = frame.to_ndarray(format="bgr24")
        
        # Converte para escala de cinza para o algoritmo
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_f32 = gray.astype(np.float32)

        # --- Algoritmo LASCA ---
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        mean_img = cv2.filter2D(img_f32, -1, kernel)
        
        # Cálculo do desvio padrão
        sq_img_mean = cv2.filter2D(img_f32**2, -1, kernel)
        std = cv2.sqrt(cv2.absdiff(sq_img_mean, mean_img**2))
        
        # Máscara de brilho mínimo
        mask = mean_img < min_gray
        
        # Cálculo do contraste (K = std/mean)
        mean_img[mean_img == 0] = 1 # Evita divisão por zero
        lasca = (std / mean_img) * (255.0 / (contrast_scale / 100.0))
        
        # Inverte e coloriza (Igual ao seu .exe)
        lasca_u8 = 255 - np.clip(lasca, 0, 255).astype(np.uint8)
        lasca_u8[mask] = 0
        
        # Aplica o mapa de cores JET
        result = cv2.applyColorMap(lasca_u8, cv2.COLORMAP_JET)
        
        return result

# O botão de "Start" que o navegador precisa para pedir permissão da câmera
webrtc_streamer(key="biospeckle", video_transformer_factory=BiospeckleTransformer)

st.write("Dica: Clique em 'Start' acima para ativar sua câmera e iniciar o processamento.")