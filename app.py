from pathlib import Path
import io
import time

import pandas as pd
import streamlit as st
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
REMEDIES_FILE = BASE_DIR / "agro_remedies.csv"
MANDI_FILE = BASE_DIR / "mandi_rates.csv"
AUDIO_DIR = BASE_DIR / "audio_cache"
AUDIO_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="KisanDrishti AI", page_icon="🌾", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
.hero {padding: 1.2rem 1.4rem; border-radius: 16px; background: linear-gradient(135deg,#eaf7ea,#f8fff5); border:1px solid #d6ead6;}
.badge {display:inline-block; padding:.25rem .55rem; border-radius:999px; background:#1b7f3a; color:white; font-size:.8rem;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🌾 KisanDrishti AI</h1><p>Edge-AI crop leaf inspection, Telugu guidance and localized APMC market intelligence for Andhra Pradesh.</p><span class="badge">Prototype • Local-first</span></div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(REMEDIES_FILE), pd.read_csv(MANDI_FILE)

remedies, mandi = load_data()

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    model_path = BASE_DIR / "yolov8n.pt"
    return YOLO(str(model_path) if model_path.exists() else "yolov8n.pt")

def telugu_audio(text: str, key: str):
    path = AUDIO_DIR / f"{key}.mp3"
    if path.exists():
        return path.read_bytes()
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="te").write_to_fp(buf)
        data = buf.getvalue()
        path.write_bytes(data)
        return data
    except Exception as exc:
        st.warning(f"Telugu audio could not be generated. gTTS needs internet for a new audio file: {exc}")
        return None

def run_inference(image: Image.Image, conf: float):
    model = load_model()
    start = time.perf_counter()
    result = model.predict(source=image, conf=conf, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    annotated = result.plot()
    labels = []
    if result.boxes is not None:
        for cls_id, score in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            labels.append((result.names[int(cls_id)], float(score)))
    return annotated, labels, elapsed_ms

tab1, tab2, tab3 = st.tabs(["🔬 Leaf Diagnosis", "📊 APMC Mandi Rates", "⚙️ Engineering Benchmarks"])

with tab1:
    st.subheader("Real-time leaf inspection")
    st.caption("Upload a leaf image or capture one from the browser camera. YOLOv8n draws bounding boxes locally.")
    source = st.radio("Image source", ["Upload image", "Camera"], horizontal=True)
    image = None
    if source == "Upload image":
        uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
    else:
        camera = st.camera_input("Capture leaf")
        if camera:
            image = Image.open(camera).convert("RGB")

    conf = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
    if image is not None and st.button("🚀 Diagnose", type="primary"):
        with st.spinner("Running local YOLOv8n inference..."):
            annotated, detections, latency = run_inference(image, conf)
        left, right = st.columns(2)
        with left:
            st.image(image, caption="Input", use_container_width=True)
        with right:
            st.image(annotated, caption="YOLOv8n bounding boxes", channels="BGR", use_container_width=True)
        st.metric("Measured inference time", f"{latency:.1f} ms")
        if detections:
            st.write("Detected labels")
            st.dataframe(pd.DataFrame(detections, columns=["Label", "Confidence"]), hide_index=True, use_container_width=True)
        else:
            st.info("No YOLOv8n object was detected above the selected confidence threshold.")

        disease = st.selectbox("Prototype remedy advisory", remedies["Disease_Name"].tolist())
        row = remedies.loc[remedies["Disease_Name"] == disease].iloc[0]
        st.markdown(f"**Crop:** {row['Crop']}")
        st.markdown(f"**Organic / cultural guidance:** {row['Organic_Remedy']}")
        st.markdown(f"**Chemical treatment:** {row['Chemical_Treatment']}")
        audio = telugu_audio(row["Telugu_Audio"], disease.lower())
        if audio:
            st.audio(audio, format="audio/mp3")
            st.caption("తెలుగు ఆడియో సలహా")

    st.warning("Prototype note: the standard yolov8n.pt checkpoint is a general object-detection model, not a plant-disease model. For production disease diagnosis, replace it with a crop-disease-trained YOLO checkpoint while keeping this interface.")

with tab2:
    st.subheader("Localized APMC market rates")
    yard = st.selectbox("APMC yard", sorted(mandi["Yard"].unique()))
    view = mandi[mandi["Yard"] == yard].copy()
    st.dataframe(view, hide_index=True, use_container_width=True)
    if not view.empty:
        st.bar_chart(view.set_index("Date")["Modal_Rs_Per_Qtl"])
    st.caption("Rates are ₹ per quintal and are a dated prototype snapshot. Verify current prices with the official AGMARKNET/data.gov.in feed before trading decisions.")

with tab3:
    st.subheader("Engineering benchmarks")
    c1, c2, c3 = st.columns(3)
    c1.metric("Target latency", "< 50 ms")
    c2.metric("Cloud dependency", "0% for inference")
    c3.metric("Model target size", "6.2 MB")
    st.markdown("### Architecture")
    st.write("• Local Streamlit UI → PIL image capture → YOLOv8n inference → bounding-box rendering")
    st.write("• CSV knowledge base for remedies and APMC rates")
    st.write("• gTTS is optional and network-dependent when generating a new Telugu audio file; cached audio can be played locally afterward.")
    st.info("The <50 ms figure is a target benchmark, not a guaranteed result. Measure on the deployment hardware with the final disease-trained model before claiming compliance.")

st.divider()
st.caption("KisanDrishti AI prototype • Andhra Pradesh • Local-first design")
