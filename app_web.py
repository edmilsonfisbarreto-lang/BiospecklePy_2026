import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="BiospecklePy", layout="wide", page_icon="🧪")

# ── CSS escuro ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  body, .stApp { background-color: #030712; color: white; }
  section[data-testid="stSidebar"] { background-color: #111827; }
  .stSlider > div { color: white; }
  h1, h2, h3, label, p { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Colormap Jet ─────────────────────────────────────────────────────────────
def jet_colormap(t):
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

# ── Processamento LASCA ───────────────────────────────────────────────────────
def process_lasca(frame, min_gray, contrast, gain, exposure):
    gain_factor = 1 + gain / 740
    exposure_factor = exposure / 5
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = gray * gain_factor * exposure_factor

    kernel_size = 5
    mean = cv2.blur(gray, (kernel_size, kernel_size))
    mean_sq = cv2.blur(gray ** 2, (kernel_size, kernel_size))
    variance = np.maximum(mean_sq - mean ** 2, 0)
    std = np.sqrt(variance)

    max_c = max(0.01, contrast / 100)
    lasca = np.where(mean > min_gray, np.clip((std / np.maximum(mean, 1e-6)) / max_c, 0, 1), 0)
    lasca_inv = 1 - lasca

    output = jet_colormap(lasca_inv)
    mask = mean < min_gray
    output[mask] = [0, 0, 0]
    return output

def process_gray(frame, gain, exposure):
    gain_factor = 1 + gain / 740
    exposure_factor = exposure / 5
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = np.clip(gray * gain_factor * exposure_factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧪 BiospecklePy")
    st.caption("Sistema de Análise LASCA")
    st.divider()

    camera_id = st.selectbox("Câmera", [0, 1, 2], format_func=lambda x: f"Câmera {x}")
    mode = st.radio("Modo de Visualização", ["FLUXO", "CINZA"])
    st.divider()

    st.markdown("**PARÂMETROS**")
    min_gray = st.slider("Mín Cinza", 0, 255, 10)
    contrast = st.slider("Contraste", 1, 100, 20)
    exposure = st.slider("Exposição", 1, 13, 5)
    gain = st.slider("Ganho", 0, 740, 0)
    zoom = st.slider("Zoom (px)", 400, 1100, 800)
    st.divider()

    capture = st.button("📸 CAPTURAR IMAGEM", use_container_width=True)
    st.divider()
    st.caption("Desenvolvido por Edmilson Souza Barreto")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("BiospecklePy — Análise LASCA em Tempo Real")

FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(camera_id)

stop = st.button("⏹ PARAR", use_container_width=True)
captured_image = None

if not cap.isOpened():
    st.error("Não foi possível acessar a câmera. Verifique se ela está conectada.")
else:
    while not stop:
        ret, frame = cap.read()
        if not ret:
            st.warning("Sem sinal da câmera.")
            break

        if mode == "FLUXO":
            result = process_lasca(frame, min_gray, contrast, gain, exposure)
        else:
            result = process_gray(frame, gain, exposure)

        display_width = min(zoom, 1000)
        display_height = int(display_width * 0.75)
        result_resized = cv2.resize(result, (display_width, display_height))

        FRAME_WINDOW.image(result_resized, channels="RGB")
        captured_image = result_resized

    cap.release()

if capture and captured_image is not None:
    img_pil = Image.fromarray(captured_image)
    import io
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    st.download_button("⬇️ Baixar Imagem", buf.getvalue(), "analise_lasca.png", "image/png")