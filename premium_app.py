
import os
import time
import base64
import html
from io import BytesIO
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image

# TensorFlow is imported lazily only when prediction is requested.
tf = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    PDF_READY = True
except Exception:
    PDF_READY = False


APP_NAME = "RetinaGuard AI"
MODEL_PATH = "models/retina_model.keras"
INPUT_SIZE = (224, 224)
MODEL_VERSION = "RetinaGuard CNN v1"

CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

DISPLAY_NAMES = {
    "No_DR": "No Diabetic Retinopathy",
    "Mild": "Mild Diabetic Retinopathy",
    "Moderate": "Moderate Diabetic Retinopathy",
    "Severe": "Severe Diabetic Retinopathy",
    "Proliferative_DR": "Proliferative Diabetic Retinopathy",
}

RISK_INFO = {
    "No_DR": {
        "level": "LOW RISK",
        "color": "#22C55E",
        "message": "No clear diabetic retinopathy signs were detected.",
        "action": "Continue routine eye screening and diabetic health monitoring.",
        "priority": "Routine",
    },
    "Mild": {
        "level": "MILD RISK",
        "color": "#EAB308",
        "message": "Early retinal changes may be present.",
        "action": "Schedule a routine ophthalmology follow-up and maintain glucose control.",
        "priority": "Follow-up",
    },
    "Moderate": {
        "level": "MODERATE RISK",
        "color": "#F97316",
        "message": "Moderate diabetic retinopathy signs may be present.",
        "action": "Book an eye specialist consultation soon.",
        "priority": "Consultation",
    },
    "Severe": {
        "level": "HIGH RISK",
        "color": "#EF4444",
        "message": "Severe retinal changes may be present.",
        "action": "Prioritize urgent ophthalmology review.",
        "priority": "Urgent",
    },
    "Proliferative_DR": {
        "level": "CRITICAL RISK",
        "color": "#E11D48",
        "message": "Advanced diabetic retinopathy signs may be present.",
        "action": "Seek immediate specialist evaluation.",
        "priority": "Immediate",
    },
}


st.set_page_config(
    page_title=APP_NAME,
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def safe(value):
    return html.escape(str(value))


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

        :root {
            --bg: #020617;
            --panel: rgba(15,23,42,0.94);
            --panel2: rgba(17,24,39,0.90);
            --line: rgba(167,139,250,0.24);
            --text: #F8FAFC;
            --muted: #CBD5E1;
            --soft: #94A3B8;
            --gold: #FDE68A;
            --blue: #3B82F6;
            --violet: #8B5CF6;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 8%, rgba(250,204,21,0.12), transparent 27%),
                radial-gradient(circle at 90% 8%, rgba(139,92,246,0.22), transparent 31%),
                radial-gradient(circle at 88% 92%, rgba(59,130,246,0.16), transparent 34%),
                linear-gradient(135deg, #020617 0%, #07111F 48%, #140A2B 100%) !important;
            color: var(--text) !important;
        }

        .block-container {
            max-width: 1220px;
            padding-top: 1rem;
            padding-bottom: 3rem;
        }

        header[data-testid="stHeader"],
        section.main,
        div[data-testid="stAppViewContainer"],
        div[data-testid="stVerticalBlock"],
        div[data-testid="stHorizontalBlock"],
        div[data-testid="stMarkdownContainer"],
        .element-container {
            background: transparent !important;
        }

        #MainMenu, footer,
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"],
        .stDeployButton {
            display: none !important;
        }

        h1, h2, h3, h4, h5, h6, p, div, span, label, li {
            color: var(--text) !important;
        }

        /* Kill white loading boards and white defaults */
        div[data-testid="stSkeleton"],
        div[data-testid="stSkeleton"] *,
        div[data-testid="stSpinner"],
        div[data-testid="stSpinner"] *,
        iframe,
        canvas,
        div[style*="background-color: rgb(255, 255, 255)"],
        div[style*="background: white"],
        div[style*="background-color: white"] {
            background: transparent !important;
            background-color: transparent !important;
            color: var(--text) !important;
            border: none !important;
            box-shadow: none !important;
        }

        /* File uploader dark mode */
        div[data-testid="stFileUploader"],
        div[data-testid="stFileUploader"] *,
        div[data-testid="stFileUploaderDropzone"],
        div[data-testid="stFileUploaderDropzone"] * {
            background-color: transparent !important;
            color: var(--text) !important;
        }

        div[data-testid="stFileUploader"] section,
        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(15,23,42,0.96) !important;
            border: 1px dashed rgba(167,139,250,0.48) !important;
            border-radius: 22px !important;
            box-shadow: none !important;
        }

        div[data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #FACC15, #A78BFA, #3B82F6) !important;
            color: #020617 !important;
            border-radius: 14px !important;
            border: none !important;
            font-weight: 900 !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            width: 100%;
            border-radius: 18px;
            padding: 0.92rem 1rem;
            font-weight: 950;
            border: none;
            color: #020617 !important;
            background: linear-gradient(135deg, #FACC15, #A78BFA, #3B82F6);
            box-shadow: 0 12px 28px rgba(139,92,246,0.28);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            color: #020617 !important;
            background: linear-gradient(135deg, #FDE68A, #C4B5FD, #60A5FA);
        }

        .app-shell {
            background: rgba(2,6,23,0.76);
            border: 1px solid rgba(167,139,250,0.20);
            border-radius: 34px;
            padding: 20px;
            box-shadow: 0 20px 80px rgba(0,0,0,0.35);
        }

        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(2,6,23,0.92);
            border: 1px solid rgba(167,139,250,0.28);
            border-radius: 28px;
            padding: 18px 24px;
            margin-bottom: 18px;
            box-shadow: 0 18px 55px rgba(0,0,0,0.42);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-logo {
            width: 54px;
            height: 54px;
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #FACC15, #A78BFA, #3B82F6);
            font-size: 25px;
            box-shadow: 0 0 34px rgba(139,92,246,0.38);
        }

        .brand-title {
            font-size: 25px;
            font-weight: 950;
            letter-spacing: -0.5px;
        }

        .brand-sub {
            color: var(--gold) !important;
            font-size: 12px;
            font-weight: 850;
            margin-top: 2px;
        }

        .status-pill {
            padding: 9px 15px;
            border-radius: 999px;
            background: rgba(250,204,21,0.10);
            border: 1px solid rgba(250,204,21,0.25);
            color: var(--gold) !important;
            font-weight: 850;
            font-size: 12px;
        }

        .nav-card {
            background: rgba(15,23,42,0.70);
            border: 1px solid rgba(167,139,250,0.18);
            border-radius: 22px;
            padding: 12px;
            margin-bottom: 20px;
        }

        .hero {
            background:
                radial-gradient(circle at top right, rgba(139,92,246,0.20), transparent 34%),
                radial-gradient(circle at bottom left, rgba(250,204,21,0.10), transparent 34%),
                linear-gradient(145deg, rgba(2,6,23,0.98), rgba(17,24,39,0.92));
            border: 1px solid rgba(167,139,250,0.30);
            border-radius: 38px;
            padding: 58px;
            box-shadow: 0 25px 85px rgba(0,0,0,0.45);
            margin-bottom: 24px;
        }

        .hero-badge {
            display: inline-block;
            border: 1px solid rgba(250,204,21,0.35);
            background: rgba(250,204,21,0.10);
            color: var(--gold) !important;
            padding: 9px 18px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 950;
            letter-spacing: 0.9px;
            margin-bottom: 20px;
        }

        .hero-title {
            font-size: 68px;
            font-weight: 950;
            line-height: 0.98;
            letter-spacing: -3px;
            margin-bottom: 20px;
        }

        .hero-title span {
            background: linear-gradient(90deg, #FDE68A, #C4B5FD, #60A5FA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-text {
            color: var(--muted) !important;
            font-size: 18px;
            line-height: 1.75;
            max-width: 820px;
            margin-bottom: 24px;
        }

        .card, .card-accent, .scan-panel, .report-panel, .doctor-panel {
            background-color: rgba(15,23,42,0.96) !important;
            border-radius: 26px;
            padding: 24px;
            height: 100%;
            box-shadow: 0 18px 45px rgba(0,0,0,0.30);
        }

        .card {
            border: 1px solid rgba(148,163,184,0.15);
        }

        .card-accent {
            background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(36,20,75,0.88)) !important;
            border: 1px solid rgba(168,85,247,0.32);
        }

        .scan-panel {
            border: 1px solid rgba(167,139,250,0.28);
        }

        .icon-box {
            width: 46px;
            height: 46px;
            border-radius: 15px;
            background: linear-gradient(135deg, #FACC15, #A78BFA, #60A5FA);
            color: #020617 !important;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 950;
            font-size: 18px;
            margin-bottom: 14px;
        }

        .card-title {
            font-size: 22px;
            font-weight: 900;
            margin-bottom: 10px;
        }

        .card-text {
            color: var(--muted) !important;
            font-size: 15px;
            line-height: 1.72;
        }

        .section-title {
            font-size: 38px;
            font-weight: 950;
            letter-spacing: -1px;
            margin-bottom: 8px;
        }

        .muted {
            color: var(--soft) !important;
            font-size: 14px;
            line-height: 1.6;
        }

        .image-card {
            background: rgba(15,23,42,0.96) !important;
            border: 1px solid rgba(167,139,250,0.24);
            border-radius: 28px;
            padding: 14px;
            box-shadow: 0 20px 55px rgba(0,0,0,0.34);
        }

        .image-card img {
            width: 100%;
            border-radius: 20px;
            display: block;
        }

        .image-caption {
            color: var(--gold) !important;
            font-size: 13px;
            font-weight: 800;
            margin-top: 10px;
            text-align: center;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin: 18px 0;
        }

        .metric-card {
            background: rgba(15,23,42,0.96) !important;
            border: 1px solid rgba(148,163,184,0.15);
            border-radius: 20px;
            padding: 16px;
        }

        .metric-label {
            color: var(--soft) !important;
            font-size: 12px;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 8px;
        }

        .metric-value {
            color: var(--text) !important;
            font-size: 16px;
            font-weight: 950;
            line-height: 1.35;
        }

        .risk-strip {
            background: rgba(15,23,42,0.96) !important;
            border: 1px solid rgba(167,139,250,0.24);
            border-radius: 26px;
            padding: 22px;
            margin-top: 16px;
        }

        .risk-badge {
            display: inline-block;
            padding: 10px 18px;
            border-radius: 999px;
            color: white !important;
            font-weight: 950;
            margin-bottom: 16px;
            letter-spacing: 0.6px;
        }

        .simple-result-row {
            margin-top: 10px;
            padding: 14px 0;
            border-bottom: 1px solid rgba(148,163,184,0.12);
        }

        .simple-result-key {
            color: var(--soft) !important;
            font-size: 12px;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }

        .simple-result-value {
            color: var(--text) !important;
            font-size: 15px;
            font-weight: 700;
            line-height: 1.6;
            margin-top: 5px;
        }

        .download-panel {
            background: linear-gradient(135deg, rgba(250,204,21,0.14), rgba(139,92,246,0.14), rgba(59,130,246,0.12)) !important;
            border: 1px solid rgba(167,139,250,0.30);
            border-radius: 24px;
            padding: 18px;
            margin: 18px 0 12px 0;
        }

        .download-title {
            font-size: 20px;
            font-weight: 950;
            margin-bottom: 6px;
        }

        .download-sub {
            color: var(--gold) !important;
            font-size: 14px;
            font-weight: 750;
        }

        .prob-wrap {
            background: rgba(15,23,42,0.96) !important;
            border: 1px solid rgba(148,163,184,0.15);
            border-radius: 24px;
            padding: 22px;
            margin-top: 16px;
        }

        .prob-row {
            margin-bottom: 15px;
        }

        .prob-label {
            display: flex;
            justify-content: space-between;
            font-weight: 800;
            font-size: 14px;
            margin-bottom: 8px;
        }

        .bar-bg {
            height: 12px;
            border-radius: 999px;
            background: rgba(148,163,184,0.18);
            overflow: hidden;
        }

        .bar-fill {
            height: 12px;
            border-radius: 999px;
            background: linear-gradient(90deg, #FACC15, #A78BFA, #3B82F6);
        }

        .dark-table {
            width: 100%;
            border-collapse: collapse;
            overflow: hidden;
            border-radius: 18px;
            background: rgba(15,23,42,0.96) !important;
            border: 1px solid rgba(167,139,250,0.22);
        }

        .dark-table th {
            background: rgba(124,58,237,0.18);
            color: var(--gold) !important;
            text-align: left;
            padding: 14px;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }

        .dark-table td {
            padding: 14px;
            border-top: 1px solid rgba(148,163,184,0.12);
            color: #E5E7EB !important;
            font-size: 14px;
        }

        .disclaimer {
            margin-top: 18px;
            padding: 18px;
            border-radius: 18px;
            background: rgba(250,204,21,0.08) !important;
            border: 1px solid rgba(250,204,21,0.22);
            color: var(--gold) !important;
            font-size: 14px;
            line-height: 1.65;
        }

        @media (max-width: 900px) {
            .workflow-grid,
            .metric-grid {
                grid-template-columns: 1fr;
            }

            .hero {
                padding: 34px;
            }

            .hero-title {
                font-size: 44px;
            }
        }

        /* FINAL: hide old clinical/risk strip blocks completely */
        .risk-strip,
        .simple-result-row,
        .result-card,
        .result-grid,
        .result-item {
            display: none !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    if tf is None:
        return None

    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(MODEL_PATH)


def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def video_to_base64(path):
    if not os.path.exists(path):
        return None

    with open(path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode()


def render_animation():
    video_base64 = video_to_base64("assets/retina_intro.mp4")

    if video_base64:
        st.markdown(
            f"""
            <div class="card-accent" style="padding:14px;">
                <video autoplay muted loop playsinline
                    style="width:100%; border-radius:22px; display:block; background:transparent;">
                    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                </video>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <div class="card-accent" style="min-height:360px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
            <div class="scanner-icon">👁️</div>
            <div class="scanner-title">AI Retina Scan</div>
            <div class="scanner-text">Add assets/retina_intro.mp4 for homepage animation.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(INPUT_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def make_prediction(model, image):
    arr = preprocess_image(image)

    start = time.time()
    probs = model.predict(arr, verbose=0)[0]
    end = time.time()

    class_index = int(np.argmax(probs))
    class_key = CLASS_NAMES[class_index]
    confidence = float(np.max(probs) * 100)
    inference_time = round((end - start) * 1000, 2)

    return class_key, confidence, probs, inference_time


def generate_ai_explanation(class_key, confidence):
    info = RISK_INFO[class_key]
    return (
        f"The uploaded retina image was classified as {DISPLAY_NAMES[class_key]} "
        f"with {confidence:.2f}% confidence. The result is mapped to {info['level']}. "
        f"{info['message']} Recommended action: {info['action']}"
    )


def create_pdf_report(class_key, confidence, probs, inference_time):
    if not PDF_READY:
        return None

    info = RISK_INFO[class_key]
    explanation = generate_ai_explanation(class_key, confidence)

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontSize=22,
        textColor=colors.HexColor("#4C1D95"),
        spaceAfter=14,
    )

    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1D4ED8"),
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#111827"),
        spaceAfter=8,
    )

    story = []
    story.append(Paragraph("RetinaGuard AI - Clinical Screening Report", title_style))
    story.append(Paragraph("AI-powered retinal assessment summary", body_style))
    story.append(Spacer(1, 12))

    summary_data = [
        ["Detected Stage", DISPLAY_NAMES[class_key]],
        ["Confidence", f"{confidence:.2f}%"],
        ["Risk Level", info["level"]],
        ["Priority", info["priority"]],
        ["Inference Time", f"{inference_time} ms"],
        ["Generated On", datetime.now().strftime("%d-%m-%Y %H:%M")],
    ]

    summary_table = Table(summary_data, colWidths=[160, 320])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#FEF3C7")),
                ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#EEF2FF")),
                ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#A78BFA")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.append(summary_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Clinical Insight", section_style))
    story.append(Paragraph(info["message"], body_style))

    story.append(Paragraph("AI Explanation", section_style))
    story.append(Paragraph(explanation, body_style))

    story.append(Paragraph("Recommended Action", section_style))
    story.append(Paragraph(info["action"], body_style))

    probability_rows = [["Class", "Probability"]]
    for i, value in enumerate(probs):
        probability_rows.append([DISPLAY_NAMES[CLASS_NAMES[i]], f"{float(value) * 100:.2f}%"])

    story.append(Paragraph("Class Probability Distribution", section_style))

    probability_table = Table(probability_rows, colWidths=[300, 180])
    probability_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7C3AED")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#A78BFA")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(probability_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Disclaimer", section_style))
    story.append(
        Paragraph(
            "This report is generated for educational and demonstration purposes only. "
            "It is not a replacement for professional medical diagnosis.",
            body_style,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def html_table(headers, rows):
    head = "".join([f"<th>{safe(h)}</th>" for h in headers])
    body = ""
    for row in rows:
        body += "<tr>" + "".join([f"<td>{safe(cell)}</td>" for cell in row]) + "</tr>"

    return f"""
    <table class="dark-table">
        <thead><tr>{head}</tr></thead>
        <tbody>{body}</tbody>
    </table>
    """


def probability_html(probs):
    rows = ""
    for i, value in enumerate(probs):
        percent = float(value) * 100
        label = safe(DISPLAY_NAMES[CLASS_NAMES[i]])

        rows += f"""
        <div class="prob-row">
            <div class="prob-label">
                <span>{label}</span>
                <span>{percent:.2f}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="width:{percent}%;"></div>
            </div>
        </div>
        """

    return f"""
    <div class="prob-wrap">
        <div class="card-title">Probability Distribution</div>
        <div class="card-text" style="margin-bottom:14px;">Class-wise model confidence</div>
        {rows}
    </div>
    """


def render_navbar():
    st.markdown(
        """
        <div class="topbar">
            <div class="brand">
                <div class="brand-logo">👁️</div>
                <div>
                    <div class="brand-title">RetinaGuard AI</div>
                    <div class="brand-sub">Clinical retinal intelligence suite</div>
                </div>
            </div>
            <div class="status-pill">AI Screening Workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_card(title, text, icon="•", accent=False):
    class_name = "card-accent" if accent else "card"

    st.markdown(
        f"""
        <div class="{class_name}">
            <div class="icon-box">{safe(icon)}</div>
            <div class="card-title">{safe(title)}</div>
            <div class="card-text">{safe(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def nav_buttons():
    st.markdown('<div class="nav-card">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        if st.button("Welcome"):
            st.session_state.page = "Welcome"
            st.rerun()

    with c2:
        if st.button("Scanner"):
            st.session_state.page = "Scanner"
            st.rerun()

    with c3:
        if st.button("Report"):
            st.session_state.page = "Report"
            st.rerun()

    with c4:
        if st.button("Doctor View"):
            st.session_state.page = "Doctor View"
            st.rerun()

    with c5:
        if st.button("Model Insights"):
            st.session_state.page = "Model Insights"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def welcome_page():
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown(
            """
            <div class="hero">
                <div class="hero-badge">WELCOME TO RETINAGUARD INTELLIGENCE</div>
                <div class="hero-title">Retina screening, risk scoring, <span>and PDF reports.</span></div>
                <div class="hero-text">
                    A premium AI-powered retinal screening dashboard for diabetic retinopathy.
                    Upload a retina image, get severity prediction, review risk guidance,
                    and download a clinical-style PDF report.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        b1, b2 = st.columns(2)

        with b1:
            if st.button("Get Started →"):
                st.session_state.page = "Scanner"
                st.rerun()

        with b2:
            if st.button("Open Report Center"):
                st.session_state.page = "Report"
                st.rerun()

    with right:
        render_animation()

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        render_card(
            "AI Screening Flow",
            "Upload retinal fundus images and receive AI-assisted severity prediction with confidence.",
            "AI",
            False,
        )

    with c2:
        render_card(
            "PDF Report Center",
            "Generate a structured report with stage, confidence, risk level, and suggested action.",
            "PDF",
            True,
        )

    with c3:
        render_card(
            "Doctor Review",
            "Simulated case-priority view for high-risk patients and follow-up workflow.",
            "DR",
            False,
        )


def scanner_placeholder():
    st.markdown(
        """
        <div class="scan-panel" style="min-height:420px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center;">
            <div class="scanner-icon">👁️</div>
            <div class="scanner-title">Retina Scanner Ready</div>
            <div class="scanner-text">
                Upload a retinal image and click analyze. The system will generate stage,
                confidence, risk level, class probabilities, and a PDF report.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_image_card(image):
    image_base64 = image_to_base64(image)

    st.markdown(
        f"""
        <div class="image-card">
            <img src="data:image/png;base64,{image_base64}" />
            <div class="image-caption">Uploaded Retina Image</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(class_key, confidence, inference_time):
    info = RISK_INFO[class_key]

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Predicted Stage</div>
                <div class="metric-value">{safe(DISPLAY_NAMES[class_key])}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Risk Level</div>
                <div class="metric-value">{safe(info["level"])}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Inference Time</div>
                <div class="metric-value">{inference_time} ms</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compact_result(class_key, confidence):
    info = RISK_INFO[class_key]
    explanation = generate_ai_explanation(class_key, confidence)

    st.markdown(
        f"""
        <div class="risk-strip">
            <div class="risk-badge" style="background:{safe(info["color"])};">{safe(info["level"])}</div>
            <div class="card-title">Scan Summary</div>

            <div class="simple-result-row">
                <div class="simple-result-key">Detected Stage</div>
                <div class="simple-result-value">{safe(DISPLAY_NAMES[class_key])}</div>
            </div>

            <div class="simple-result-row">
                <div class="simple-result-key">Clinical Insight</div>
                <div class="simple-result-value">{safe(info["message"])}</div>
            </div>

            <div class="simple-result-row">
                <div class="simple-result-key">Recommended Action</div>
                <div class="simple-result-value">{safe(info["action"])}</div>
            </div>

            <div class="simple-result-row" style="border-bottom:none;">
                <div class="simple-result-key">AI Explanation</div>
                <div class="simple-result-value">{safe(explanation)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pdf_download(class_key, confidence, probs, inference_time):
    pdf_bytes = create_pdf_report(class_key, confidence, probs, inference_time)

    st.markdown(
        """
        <div class="download-panel">
            <div class="download-title">PDF Report Ready</div>
            <div class="download-sub">Download the generated clinical screening summary.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if pdf_bytes:
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="retinaguard_report.pdf",
            mime="application/pdf",
        )
    else:
        st.markdown(
            '<div class="disclaimer">PDF package missing. Run: pip install reportlab</div>',
            unsafe_allow_html=True,
        )


def scanner_page():
    st.markdown('<div class="section-title">Retina Scanner</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Upload a retina image and generate a complete screening result.</div>', unsafe_allow_html=True)

    model = load_model()

    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        render_card(
            "Upload Retina Image",
            "Supported file types: JPG, JPEG, PNG. Use a clear retinal fundus image for best demo result.",
            "U",
            True,
        )

        uploaded_file = st.file_uploader(
            "Choose retina image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        analyze = st.button("Analyze Retina Image")

        st.markdown("<br>", unsafe_allow_html=True)

        render_card(
            "Report Output",
            "After analysis, the app will generate prediction, confidence, risk level, class probabilities, and a PDF download.",
            "PDF",
            False,
        )

    with right:
        if uploaded_file is None:
            scanner_placeholder()
            return

        image = Image.open(uploaded_file).convert("RGB")
        render_image_card(image)

        if not analyze:
            return

        if model is None:
            error_msg = st.session_state.get("model_load_error", "Model file missing or TensorFlow could not load.")
            st.markdown(
                f'<div class="disclaimer">Model not ready: {safe(error_msg)}</div>',
                unsafe_allow_html=True,
            )
            return

        class_key, confidence, probs, inference_time = make_prediction(model, image)

        st.session_state.last_result = {
            "class_key": class_key,
            "confidence": confidence,
            "probs": probs.tolist(),
            "inference_time": inference_time,
            "image_base64": image_to_base64(image),
            "time": datetime.now().strftime("%d-%m-%Y %H:%M"),
        }

        render_metric_grid(class_key, confidence, inference_time)
        render_pdf_download(class_key, confidence, probs, inference_time)
        render_card(
            "Latest PDF Summary",
            f"{DISPLAY_NAMES[class_key]} detected with {confidence:.2f}% confidence. Download the PDF report above.",
            "PDF",
            True,
        )
        render_card(
            "Report Generated",
            f"{DISPLAY_NAMES[class_key]} detected with {confidence:.2f}% confidence. Open the PDF for full clinical-style summary.",
            "✓",
            True,
        )
        st.markdown(probability_html(probs), unsafe_allow_html=True)

        st.markdown(
            """
            <div class="disclaimer">
                Educational and hackathon demo only. This is not a replacement for professional medical diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )


def report_page():
    st.markdown('<div class="section-title">Report Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Review the latest generated scan result and download the PDF report.</div>', unsafe_allow_html=True)

    if "last_result" not in st.session_state:
        st.markdown(
            """
            <div class="scan-panel" style="min-height:420px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center;">
                <div class="scanner-icon">📄</div>
                <div class="scanner-title">No Report Generated Yet</div>
                <div class="scanner-text">
                    Go to Scanner, upload a retina image, and analyze it.
                    The generated report will appear here with a PDF download option.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    result = st.session_state.last_result
    class_key = result["class_key"]
    confidence = result["confidence"]
    probs = np.array(result["probs"])
    inference_time = result["inference_time"]
    info = RISK_INFO[class_key]

    c1, c2 = st.columns([0.85, 1.15], gap="large")

    with c1:
        st.markdown(
            f"""
            <div class="image-card">
                <img src="data:image/png;base64,{result["image_base64"]}" />
                <div class="image-caption">Last Analyzed Retina Image</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        render_metric_grid(class_key, confidence, inference_time)
        render_pdf_download(class_key, confidence, probs, inference_time)

    st.markdown("<br>", unsafe_allow_html=True)

    rows = [
        ["Generated On", result["time"]],
        ["Stage", DISPLAY_NAMES[class_key]],
        ["Risk Level", info["level"]],
        ["Priority", info["priority"]],
        ["Recommended Action", info["action"]],
    ]

    st.markdown(html_table(["Report Field", "Value"], rows), unsafe_allow_html=True)


def doctor_page():
    st.markdown('<div class="section-title">Doctor Review</div>', unsafe_allow_html=True)

    rows = [
        ["RG-1001", "Proliferative DR", "Critical Risk", "Immediate review", "Flagged"],
        ["RG-1002", "Moderate DR", "Moderate Risk", "Specialist consult", "Follow-up"],
        ["RG-1003", "No DR", "Low Risk", "Routine check", "Stable"],
        ["RG-1004", "Severe DR", "High Risk", "Priority evaluation", "Flagged"],
    ]

    st.markdown(
        html_table(
            ["Patient ID", "Predicted Stage", "Risk Level", "Action", "Status"],
            rows,
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)

    with a1:
        if st.button("Send Review Alert"):
            st.markdown('<div class="disclaimer">Demo review alert created.</div>', unsafe_allow_html=True)
    with a2:
        if st.button("Notify Coordinator"):
            st.markdown('<div class="disclaimer">Coordinator notification created.</div>', unsafe_allow_html=True)
    with a3:
        if st.button("Mark Follow-up"):
            st.markdown('<div class="disclaimer">Case marked for follow-up.</div>', unsafe_allow_html=True)


def insights_page():
    st.markdown('<div class="section-title">Model Insights</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        render_card(
            "Model Task",
            "Five-class retinal image classification for diabetic retinopathy severity screening.",
            "M",
            False,
        )

    with c2:
        render_card(
            "Input Format",
            "The uploaded image is converted to RGB and resized to 224 × 224 before prediction.",
            "I",
            True,
        )

    with c3:
        render_card(
            "Output",
            "The model returns class probabilities, confidence score, risk level, and PDF report.",
            "O",
            False,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    rows = [
        ["No_DR", "No diabetic retinopathy"],
        ["Mild", "Early diabetic retinopathy stage"],
        ["Moderate", "Moderate disease stage"],
        ["Severe", "High-risk disease stage"],
        ["Proliferative_DR", "Advanced disease stage"],
    ]

    st.markdown(html_table(["Class", "Meaning"], rows), unsafe_allow_html=True)


def main():
    inject_css()
    render_navbar()

    if "page" not in st.session_state:
        st.session_state.page = "Welcome"

    nav_buttons()

    if st.session_state.page == "Welcome":
        welcome_page()
    elif st.session_state.page == "Scanner":
        scanner_page()
    elif st.session_state.page == "Report":
        report_page()
    elif st.session_state.page == "Doctor View":
        doctor_page()
    elif st.session_state.page == "Model Insights":
        insights_page()


if __name__ == "__main__":
    main()
