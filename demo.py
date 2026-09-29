import streamlit as st
import cv2 as cv
import numpy as np
import keras

# ---------------- Page meta ----------------
st.set_page_config(
    page_title="Grape Leaf Disease Detection",
    page_icon="🍇",
    layout="centered"
)

# ---------------- Global HTML/CSS theming ----------------
st.markdown("""
<style>
/* --- Base layout --- */
.stApp {
  background: radial-gradient(1200px 600px at 10% 10%, rgba(147, 197, 253, 0.25), transparent 60%),
              radial-gradient(1000px 500px at 90% 0%, rgba(167, 243, 208, 0.22), transparent 60%),
              linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
  color: #e5e7eb !important;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji" !important;
}

/* Center content and add a 'card' */
.main .block-container {
  max-width: 860px !important;
  padding-top: 0 !important;
}

.glass-card {
  background: rgba(15, 23, 42, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.15);
  box-shadow: 0 10px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
  backdrop-filter: blur(10px);
  border-radius: 18px;
  padding: 28px 28px 18px 28px;
  margin: 18px 0 24px 0;
}

/* Header banner */
.hero {
  border-radius: 22px;
  padding: 28px 24px;
  margin: 20px 0 14px 0;
  background:
    radial-gradient(600px 220px at 5% 0%, rgba(59,130,246,0.2), transparent 60%),
    radial-gradient(600px 220px at 95% 0%, rgba(52,211,153,0.2), transparent 60%),
    linear-gradient(180deg, rgba(30,41,59,0.7), rgba(15,23,42,0.6));
  border: 1px solid rgba(148,163,184,0.18);
  box-shadow: 0 8px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
}

.hero h1 {
  margin: 0 0 6px 0 !important;
  font-size: 30px !important;
  letter-spacing: 0.2px;
  color: #f8fafc;
}

.hero p {
  margin: 0 !important;
  color: #cbd5e1 !important;
  font-size: 15px !important;
}

/* Typography */
h2, h3 { color: #e2e8f0 !important; }
p, li, .stMarkdown, .stText, .stWrite { color: #cbd5e1 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
  background: rgba(2,6,23,0.35);
  border: 1px dashed rgba(148,163,184,0.35);
  border-radius: 16px;
  padding: 18px;
}
[data-testid="stFileUploader"] label {
  color: #e5e7eb !important;
  font-weight: 600;
}
[data-testid="stFileUploader"] section > div {
  color: #94a3b8 !important;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(180deg, #3b82f6, #2563eb);
  color: white;
  border: 0;
  border-radius: 12px;
  padding: 10px 16px;
  font-weight: 600;
  transition: transform .04s ease;
  box-shadow: 0 6px 16px rgba(37,99,235,.35);
}
.stButton > button:hover { transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

/* Image frame */
.stImage > img, .stImage img {
  border-radius: 14px;
  border: 1px solid rgba(148,163,184,0.2);
  box-shadow: 0 10px 24px rgba(0,0,0,0.35);
}

/* Footer */
.footer {
  opacity: .9;
  margin-top: 10px;
  padding: 10px 0 30px 0;
  color: #94a3b8;
  font-size: 13px;
  text-align: center;
}
.footer a { color: #c7d2fe; text-decoration: none; border-bottom: 1px dotted rgba(199,210,254,.6); }
.footer a:hover { opacity: .9; }
</style>

<div class="hero">
  <h1>🍇 Grape Leaf Disease Detection</h1>
  <p>Upload a grape leaf image to get an instant prediction powered by transfer learning.</p>
</div>
""", unsafe_allow_html=True)

# Open the glass card wrapper
st.markdown('<div class="glass-card">The grape leaf disease detection model is built using deep learning techniques, and it uses transfer learning to leverage the pre-trained knowledge of a base model. The model is trained on a dataset containing images of 33 different types of leaf diseases. For more information about the architecture, dataset, and training process, please refer to the code and documentation provided.', unsafe_allow_html=True)

# ---------------- App content ----------------
label_name = ['Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy']

# st.write("""The grape leaf disease detection model is built using deep learning techniques, and it uses transfer learning to leverage the pre-trained knowledge of a base model. The model is trained on a dataset containing images of 33 different types of leaf diseases. For more information about the architecture, dataset, and training process, please refer to the code and documentation provided.""")

st.write("Please input only leaf Images of Grape Leaves. Otherwise, the model will not work perfectly.")

# Always render the uploader first (keeps page functional even if model load fails)
uploaded_file = st.file_uploader("Upload an image")

# Lazy-load the model only when needed (with guard)
model = None
model_path = r'D:\Projects\leaf-diseases-detect-main\Training\model\Leaf Deases(96,88).h5'  # raw string for Windows path

if uploaded_file is not None:
    try:
        if model is None:
            model = keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Model failed to load from:\n{model_path}\n\nError: {e}")
        st.markdown('</div><div class="footer">Made with ❤️ for smart farming • <a href="#">Docs</a></div>', unsafe_allow_html=True)
        st.stop()

    # Read and preprocess image
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
    normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)

    # Predict
    with st.spinner("Analyzing image..."):
        predictions = model.predict(normalized_image)

    # Show uploaded image
    st.image(image_bytes, caption="Uploaded image", use_container_width=True)

    # Compute result
    pred = predictions[0]
    idx = int(np.argmax(pred))
    confidence = float(pred[idx]) * 100.0

    # Result box (clear + visible)
    if confidence >= 80:
        if idx < len(label_name):
            st.markdown(
                f"""
                <div style="
                    margin-top:12px;
                    padding:14px 16px;
                    border-radius:12px;
                    border:1px solid rgba(34,197,94,0.35);
                    background: rgba(34,197,94,0.12);
                    font-weight:600;">
                    Result: {label_name[idx]} ({confidence:.2f}%)
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.warning(f"Error: Prediction index {idx} out of range.")
    else:
        st.info("Confidence below 80%. Try another image.")

# Close the glass card and add footer
st.markdown('</div><div class="footer">Made for smart farming • <a href="#">Docs</a></div>', unsafe_allow_html=True)
