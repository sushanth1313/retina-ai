import os
import numpy as np
import streamlit as st
from PIL import Image

try:
    import tensorflow as tf
except Exception:
    tf = None


APP_NAME = "RetinaGuard AI"
MODEL_PATH = "models/retina_model.keras"

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]


st.set_page_config(
    page_title=APP_NAME,
    page_icon="👁️",
    layout="wide"
)


def load_model():
    if tf is None:
        return None

    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def get_risk_message(predicted_class):
    if predicted_class == 0:
        return "Low Risk", "No clear signs of diabetic retinopathy detected."
    elif predicted_class == 1:
        return "Mild Risk", "Early signs detected. Regular eye check-up is recommended."
    elif predicted_class == 2:
        return "Moderate Risk", "Noticeable diabetic retinopathy signs detected. Medical consultation is recommended."
    elif predicted_class == 3:
        return "High Risk", "Severe signs detected. Please consult an eye specialist soon."
    else:
        return "Critical Risk", "Advanced signs detected. Immediate medical attention is recommended."


model = load_model()


st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #050816, #0f172a);
    }
    .title {
        font-size: 48px;
        font-weight: 800;
        color: #38bdf8;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 20px;
        color: #cbd5e1;
        text-align: center;
        margin-bottom: 35px;
    }
    .card {
        background: rgba(15, 23, 42, 0.9);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(56, 189, 248, 0.3);
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.15);
    }
    .warning {
        background: rgba(251, 191, 36, 0.12);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(251, 191, 36, 0.4);
        color: #fde68a;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(f"<div class='title'>👁️ {APP_NAME}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>AI-powered diabetic retinopathy screening from retina images</div>",
    unsafe_allow_html=True
)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("### Upload Retina Image")
    uploaded_file = st.file_uploader(
        "Choose a retina image",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown(
        """
        <div class='warning'>
        ⚠️ This app is for educational and hackathon demo purposes only.
        It is not a replacement for professional medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True
    )

with right_col:
    st.markdown("### AI Result")

    if uploaded_file is None:
        st.info("Upload a retina image to get prediction.")

    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Retina Image", use_container_width=True)

        if model is None:
            st.error("Model not found yet. Train the model first to create models/retina_model.keras")
        else:
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)

            predicted_class = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0]) * 100)

            risk_level, message = get_risk_message(predicted_class)

            st.success("Prediction completed")

            st.metric("Detected Stage", CLASS_NAMES[predicted_class])
            st.metric("Confidence", f"{confidence:.2f}%")
            st.metric("Risk Level", risk_level)

            st.write(message)

            report = f"""
RetinaGuard AI Report

Prediction: {CLASS_NAMES[predicted_class]}
Confidence: {confidence:.2f}%
Risk Level: {risk_level}

Message:
{message}

Disclaimer:
This report is generated for educational and hackathon demo purposes only.
It is not a replacement for professional medical diagnosis.
"""

            st.download_button(
                label="Download Report",
                data=report,
                file_name="retinaguard_report.txt",
                mime="text/plain"
            )


st.markdown("---")
st.markdown("### Project Pipeline")
st.write(
    "Retina Image Upload → Image Preprocessing → CNN Model Prediction → Risk Level → Report"
)