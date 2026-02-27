import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="BiospecklePy Web", layout="wide")

st.title("🌱 BiospecklePy - Análise Online")
st.sidebar.header("Configurações do Algoritmo")

# Upload do arquivo
uploaded_file = st.file_uploader("Suba um vídeo ou imagem de Speckle", type=['mp4', 'avi', 'png', 'jpg', 'tif'])

# Controles na barra lateral
kernel_size = st.sidebar.slider("Tamanho do Kernel", 3, 15, 5, step=2)
contrast_gain = st.sidebar.slider("Ganho de Contraste", 1.0, 10.0, 2.0)

if uploaded_file is not None:
    # Lógica simplificada para processar a imagem carregada
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        img_f32 = img.astype(np.float32)
        
        # Algoritmo LASCA simplificado
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        mean = cv2.filter2D(img_f32, -1, kernel)
        sq_mean = cv2.filter2D(img_f32**2, -1, kernel)
        std = cv2.sqrt(cv2.absdiff(sq_mean, mean**2))
        
        # Evita divisão por zero
        mean[mean == 0] = 1
        contrast = std / mean
        
        # Normalização para visualização
        flow = 255 - (contrast * (255 / contrast_gain))
        flow = np.clip(flow, 0, 255).astype(np.uint8)
        flow_color = cv2.applyColorMap(flow, cv2.COLORMAP_JET)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original (Speckle)", use_column_width=True)
        with col2:
            st.image(flow_color, caption="Mapa de Fluxo (LASCA)", use_column_width=True)