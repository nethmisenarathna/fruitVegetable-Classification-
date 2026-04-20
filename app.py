import streamlit as st
import json
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    page_icon="🥕",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=DM+Serif+Display:ital@0;1&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 5% 10%, rgba(255, 187, 92, 0.35), transparent 34%),
            radial-gradient(circle at 95% 15%, rgba(78, 163, 120, 0.30), transparent 36%),
            linear-gradient(120deg, #fff9f0 0%, #f6fffb 50%, #fff6f4 100%);
        color: #1f2a26;
        font-family: 'Space Grotesk', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
        letter-spacing: 0.2px;
        color: #122019;
    }

    .hero {
        background: linear-gradient(120deg, rgba(255, 179, 71, 0.90), rgba(74, 157, 110, 0.90));
        border-radius: 20px;
        padding: 1.5rem 1.6rem;
        color: #fff;
        box-shadow: 0 14px 30px rgba(39, 73, 54, 0.22);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        color: #fff;
        font-size: 2rem;
    }

    .hero p {
        margin: 0.5rem 0 0;
        font-size: 1.03rem;
        opacity: 0.95;
    }

    .panel {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(255, 255, 255, 0.65);
        border-radius: 18px;
        padding: 1rem;
        backdrop-filter: blur(8px);
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.08);
    }

    .result-badge {
        background: linear-gradient(120deg, #1f8f5f, #f2a65a);
        color: #fff;
        padding: 0.65rem 1rem;
        border-radius: 12px;
        display: inline-block;
        font-size: 1.08rem;
        font-weight: 700;
    }

    .note {
        font-size: 0.92rem;
        opacity: 0.8;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #1f2a26 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #2f7f59 !important;
    }

    .stTabs [data-baseweb="tab-list"] button {
        color: #1f2a26 !important;
        opacity: 1 !important;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #0f6d48 !important;
        border-bottom: 2px solid #0f6d48 !important;
    }

    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #145a3c !important;
    }

    [data-testid="stFileUploader"] section button {
        background-color: #0f6d48 !important;
        color: #ffffff !important;
        border: 1px solid #0b5638 !important;
        font-weight: 600;
    }

    [data-testid="stFileUploader"] section button:hover {
        background-color: #0b5638 !important;
        color: #ffffff !important;
    }

    [data-testid="stFileUploader"] small {
        color: #1f2a26 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

# Load model
model = load_model("fruitveg_resnet_realistic.keras")

st.markdown(
    """
    <div class='hero'>
      <h1>Fruit & Vegetable Classifier</h1>
      <p>Upload a produce image and get an instant AI prediction with confidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### About")
st.sidebar.write(
    "This app uses a trained ResNet model to identify fruits and vegetables from uploaded images."
)
st.sidebar.markdown("### Tips")
st.sidebar.write("Use a clear, front-facing image with good lighting for best results.")

st.markdown("### Input Source")
scan_tab, upload_tab = st.tabs(["scan from camera", "upload image"])

captured_file = None
uploaded_file = None

with scan_tab:
    st.markdown("Use your webcam to capture a fresh image.")
    _, camera_col, _ = st.columns([1, 1.6, 1])
    with camera_col:
        captured_file = st.camera_input("Open camera")

with upload_tab:
    st.markdown("Choose an image file from your device.")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

input_file = captured_file if captured_file is not None else uploaded_file
source_label = "Scanned Image" if captured_file is not None else "Uploaded Image"

if input_file:
    left_col, right_col = st.columns([1.1, 1], gap="large")

    image = Image.open(input_file).convert("RGB")

    with left_col:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader(source_label)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    img = image.resize((224, 224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img, verbose=0)
    class_id = int(np.argmax(prediction))
    confidence = float(prediction[0][class_id])
    predicted_label = class_names[class_id].title()

    with right_col:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Prediction Result")
        st.markdown(
            f"<div class='result-badge'>Predicted: {predicted_label}</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.progress(min(max(confidence, 0.0), 1.0))
        st.markdown(
            "<p class='note'>Confidence is model certainty for the selected class.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Capture from camera or upload a JPG/PNG image to start classification.")