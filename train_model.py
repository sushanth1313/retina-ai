import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import tensorflow as tf
except Exception:
    tf = None


APP_NAME = "RetinaGuard AI"
MODEL_PATH = "models/retina_model.keras"

CLASS_NAMES = [
    "No_DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative_DR"
]

DISPLAY_NAMES = {
    "No_DR": "No Diabetic Retinopathy",
    "Mild": "Mild Diabetic Retinopathy",
    "Moderate": "Moderate Diabetic Retinopathy",
    "Severe": "Severe Diabetic Retinopathy",
    "Proliferative_DR": "Proliferative Diabetic Retinopathy"
}

RISK_DETAILS = {
    "No_DR": {
        "level": "Low Risk",
        "tag_class": "tag-low",
        "message": "No clear diabetic retinopathy signs detected from the uploaded retina image.",
        "recommendation": [
            "Maintain regular eye checkups.",
            "Continue healthy diabetic management.",
            "Repeat screening as advised by a doctor."
        ]
    },
    "Mild": {
        "level": "Mild Risk",
        "tag_class": "tag-mild",
        "message": "Early-stage retinal changes may be present. A preventive consultation is recommended.",
        "recommendation": [
            "Consult an eye specialist for routine follow-up.",
            "Maintain blood sugar control carefully.",
            "Monitor symptoms and repeat retinal screening."
        ]
    },
    "Moderate": {
        "level": "Moderate Risk",
        "tag_class": "tag-moderate",
        "message": "Moderate diabetic retinopathy signs may be present and should be medically reviewed.",
        "recommendation": [
            "Book a specialist consultation soon.",
            "Follow diabetes, BP, and lifestyle management carefully.",
            "Do not delay retina follow-up."
        ]
    },
    "Severe": {
        "level": "High Risk",
        "tag_class": "tag-severe",
        "message": "Severe retinal changes may be present. Prompt specialist attention is recommended.",
        "recommendation": [
            "Consult an ophthalmologist urgently.",
            "Prioritize immediate retina evaluation.",
            "Follow professional treatment advice."
        ]
    },
    "Proliferative_DR": {
        "level": "Critical Risk",
        "tag_class": "tag-critical",
        "message": "Advanced signs may be present. Immediate specialist evaluation is strongly recommended.",
        "recommendation": [
            "Seek urgent ophthalmology consultation.",
            "Do not ignore symptoms or delay care.",
            "Proceed with immediate medical review."
        ]
    }
}


st.set_page_config(
    page_title=APP_NAME,
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56,189,248,0.18), transparent 25%),
                radial-gradient(circle at top right, rgba(59,130,246,0.14), transparent 20%),
                linear-gradient(135deg, #020617 0%, #081126 35%, #0b1120 65%, #020617 100%);
            color: #e2e8f0;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }

        h1, h2, h3, h4, h5, h6, p, div, span, label {
            color: #e2e8f0;
        }

        .hero-box {
            background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(30,41,59,0.82));
            border: 1px solid rgba(56,189,248,0.30);
            box-shadow: 0 0 40px rgba(56,189,248,0.10);
            border-radius: 28px;
            padding: 42px 36px;
            margin-bottom: 24px;
        }

        .hero-title {
            font-size: 56px;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 12px;
            color: #f8fafc;
        }

        .hero-title span {
            color: #38bdf8;
        }

        .hero-subtitle {
            font-size: 20px;
            color: #cbd5e1;
            line-height: 1.7;
            margin-bottom: 18px;
        }

        .mini-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.4px;
            background: rgba(56,189,248,0.14);
            color: #7dd3fc;
            border: 1px solid rgba(56,189,248,0.35);
            margin-bottom: 16px;
        }

        .glass-card {
            background: rgba(15,23,42,0.78);
            border: 1px solid rgba(148,163,184,0.15);
            box-shadow: 0 0 24px rgba(15,23,42,0.28);
            border-radius: 22px;
            padding: 24px;
            height: 100%;
        }

        .section-title {
            font-size: 28px;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 16px;
        }

        .feature-title {
            font-size: 20px;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 8px;
        }

        .feature-text {
            color: #cbd5e1;
            line-height: 1.65;
            font-size: 15px;
        }

        .metric-box {
            background: linear-gradient(180deg, rgba(15,23,42,0.90), rgba(15,23,42,0.65));
            border: 1px solid rgba(56,189,248,0.18);
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 12px;
        }

        .metric-label {
            color: #94a3b8;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            margin-bottom: 6px;
        }

        .metric-value {
            color: #f8fafc;
            font-size: 23px;
            font-weight: 800;
        }

        .tag {
            display: inline-block;
            padding: 9px 16px;
            border-radius: 999px;
            font-size: 14px;
            font-weight: 800;
            margin-top: 6px;
            margin-bottom: 10px;
            letter-spacing: 0.3px;
        }

        .tag-low {
            background: rgba(34,197,94,0.16);
            color: #86efac;
            border: 1px solid rgba(34,197,94,0.35);
        }

        .tag-mild {
            background: rgba(250,204,21,0.14);
            color: #fde68a;
            border: 1px solid rgba(250,204,21,0.35);
        }

        .tag-moderate {
            background: rgba(251,146,60,0.14);
            color: #fdba74;
            border: 1px solid rgba(251,146,60,0.35);
        }

        .tag-severe {
            background: rgba(248,113,113,0.14);
            color: #fca5a5;
            border: 1px solid rgba(248,113,113,0.35);
        }

        .tag-critical {
            background: rgba(244,63,94,0.14);
            color: #fda4af;
            border: 1px solid rgba(244,63,94,0.35);
        }

        .report-box {
            background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.82));
            border: 1px solid rgba(56,189,248,0.18);
            border-radius: 22px;
            padding: 26px;
            margin-top: 14px;
        }

        .report-title {
            font-size: 26px;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 12px;
        }

        .report-text {
            color: #cbd5e1;
            font-size: 15px;
            line-height: 1.75;
        }

        .small-note {
            color: #94a3b8;
            font-size: 13px;
            line-height: 1.6;
        }

        .footer-line {
            color: #94a3b8;
            text-align: center;
            font-size: 13px;
            margin-top: 28px;
        }

        .stButton > button {
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(56,189,248,0.35);
            background: linear-gradient(135deg, #0ea5e9, #2563eb);
            color: white;
            font-weight: 700;
            padding: 0.75rem 1rem;
            box-shadow: 0 8px 22px rgba(37,99,235,0.22);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #38bdf8, #1d4ed8);
            color: white;
            border: 1px solid rgba(125,211,252,0.45);
        }

        div[data-testid="stFileUploader"] > section {
            background: rgba(15,23,42,0.55);
            border: 1px dashed rgba(56,189,248,0.40);
            border-radius: 18px;
            padding: 8px;
        }

        div[data-testid="stMetric"] {
            background: rgba(15,23,42,0.52);
            border: 1px solid rgba(148,163,184,0.12);
            padding: 12px;
            border-radius: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


@st.cache_resource
def load_model():
    if tf is None:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_image(model, image):
    processed = preprocess_image(image)
    probs = model.predict(processed, verbose=0)[0]
    idx = int(np.argmax(probs))
    class_key = CLASS_NAMES[idx]
    return class_key, probs


def build_report_text(class_key, confidence, probs):
    info = RISK_DETAILS[class_key]

    probability_lines = []
    for i, score in enumerate(probs):
        class_name = DISPLAY_NAMES[CLASS_NAMES[i]]
        probability_lines.append(f"{class_name}: {score * 100:.2f}%")

    recommendations = "\n".join([f"- {item}" for item in info["recommendation"]])

    report = f"""
RetinaGuard AI — Clinical Screening Report

Prediction:
{DISPLAY_NAMES[class_key]}

Confidence:
{confidence:.2f}%

Risk Level:
{info["level"]}

Clinical Insight:
{info["message"]}

Class Probabilities:
{chr(10).join(probability_lines)}

Recommended Next Steps:
{recommendations}

Disclaimer:
This report is generated for educational and hackathon demo purposes only.
It is not a replacement for professional medical diagnosis.
""".strip()

    return report


def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()


def render_home():
    st.markdown(
        """
        <div class="hero-box">
            <div class="mini-badge">AI • Healthcare • Retinal Risk Screening</div>
            <div class="hero-title">Welcome to <span>RetinaGuard AI</span></div>
            <div class="hero-subtitle">
                A smart diabetic retinopathy screening platform that analyzes retina images,
                predicts severity stage, and generates a clean decision-ready report for fast demo presentation.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="feature-title">⚡ Instant AI Screening</div>
                <div class="feature-text">
                    Upload a retina image and get severity-stage prediction in seconds using a trained deep learning model.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="feature-title">📊 Premium Report Output</div>
                <div class="feature-text">
                    View confidence, class probabilities, risk level, medical insight, and actionable recommendations in one place.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="glass-card">
                <div class="feature-title">🏆 Hackathon-Ready UX</div>
                <div class="feature-text">
                    Designed with a futuristic medical-AI interface, clean storytelling flow, and strong presentation appeal.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns([1, 1.2, 1])
    with b2:
        if st.button("🚀 Launch Retina Scanner"):
            go_to("scanner")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Why This Project Stands Out</div>
                <div class="feature-text">
                    RetinaGuard AI combines medical imaging, explainable prediction output, and an elegant UI into one compact prototype.
                    It demonstrates how AI can assist early risk screening for diabetic retinopathy while still keeping a clear
                    educational and ethical disclaimer.
                </div>
                <br>
                <div class="feature-text">
                    <b>Project Flow:</b><br>
                    Retina Image Upload → Image Preprocessing → CNN Prediction → Risk Analysis → Styled Report
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Core Highlights</div>
                <div class="feature-text">
                    ✅ Deep learning based retina-stage classification<br><br>
                    ✅ Clean confidence and probability visualization<br><br>
                    ✅ Immediate risk interpretation for presentation<br><br>
                    ✅ Downloadable report for demo and evaluation
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="footer-line">
            Educational / hackathon prototype • Not a substitute for professional diagnosis
        </div>
        """,
        unsafe_allow_html=True
    )


def render_scanner():
    model = load_model()

    top1, top2, top3 = st.columns([4, 1, 1])
    with top1:
        st.markdown(
            """
            <div class="hero-box" style="padding: 28px 28px; margin-bottom: 18px;">
                <div class="mini-badge">Retina Screening Console</div>
                <div class="hero-title" style="font-size: 42px;">AI Retina Analysis Dashboard</div>
                <div class="hero-subtitle" style="font-size:18px; margin-bottom:0;">
                    Upload a retinal scan to generate prediction, risk level, and a polished clinical-style report.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with top2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🏠 Home"):
            go_to("home")
    with top3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset"):
            st.rerun()

    left_col, right_col = st.columns([0.95, 1.05], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Upload Retina Image</div>
                <div class="small-note">
                    Supported formats: JPG, JPEG, PNG<br>
                    Best results come from clear retinal fundus images.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose a retina image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">How to Present This</div>
                <div class="feature-text">
                    1. Upload a retina image<br>
                    2. Show predicted stage and risk level<br>
                    3. Explain confidence and class probability chart<br>
                    4. Present the recommendation block as the final report insight
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right_col:
        if uploaded_file is None:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Awaiting Scan</div>
                    <div class="feature-text">
                        Upload a retina image to generate the AI-powered report.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Retina Image", use_container_width=True)

            if model is None:
                st.error("Model not found. Please make sure models/retina_model.keras exists.")
                return

            class_key, probs = predict_image(model, image)
            confidence = float(np.max(probs) * 100)
            info = RISK_DETAILS[class_key]

            st.markdown("<br>", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Detected Stage", DISPLAY_NAMES[class_key])
            with m2:
                st.metric("Confidence", f"{confidence:.2f}%")
            with m3:
                st.metric("Risk Level", info["level"])

            st.markdown(
                f"""
                <div class="report-box">
                    <div class="report-title">Clinical Insight</div>
                    <div class="tag {info["tag_class"]}">{info["level"]}</div>
                    <div class="report-text">
                        <b>Prediction:</b> {DISPLAY_NAMES[class_key]}<br><br>
                        <b>Interpretation:</b> {info["message"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                "Stage": [DISPLAY_NAMES[name] for name in CLASS_NAMES],
                "Probability": [float(x) * 100 for x in probs]
            })

            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Class Probability Distribution</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.bar_chart(prob_df.set_index("Stage"))

            recommendations_html = "".join(
                [f"<li>{item}</li>" for item in info["recommendation"]]
            )

            st.markdown(
                f"""
                <div class="report-box">
                    <div class="report-title">Recommended Next Steps</div>
                    <div class="report-text">
                        <ul>
                            {recommendations_html}
                        </ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="report-box">
                    <div class="report-title">Disclaimer</div>
                    <div class="report-text">
                        This system is intended for educational and hackathon demonstration purposes only.
                        It does not replace professional ophthalmology diagnosis or treatment.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            report_text = build_report_text(class_key, confidence, probs)

            st.download_button(
                label="📄 Download Report",
                data=report_text,
                file_name="retinaguard_report.txt",
                mime="text/plain"
            )


def main():
    inject_css()

    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        render_home()
    else:
        render_scanner()


if __name__ == "__main__":
    main()