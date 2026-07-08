import os
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="RetinaScan AI",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else "🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CONSTANTS
# =========================================================
IMG_SIZE = 384
DISEASE_ORDER = ["DR", "MH", "ODC", "TSLN", "DN", "MYA", "ARMD", "BRVO", "ODP", "ODE"]
DISEASE_NAMES = {
    "DR": "Diabetic Retinopathy",
    "MH": "Media Haze",
    "ODC": "Optic Disc Cupping",
    "TSLN": "Tessellation",
    "DN": "Drusen",
    "MYA": "Myopia",
    "ARMD": "Age-Related Macular Degeneration",
    "BRVO": "Branch Retinal Vein Occlusion",
    "ODP": "Optic Disc Pallor",
    "ODE": "Optic Disc Edema",
}
PRIMARY = "#2563eb"
PRIMARY_DARK = "#1d4ed8"
RED = "#dc2626"
AMBER = "#d97706"
GREEN = "#16a34a"
SLATE_BG = "#f8fafc"
CARD_BG = "#ffffff"
BORDER = "#e2e8f0"
TEXT_MAIN = "#0f172a"
TEXT_MUTED = "#64748b"

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
}}

.stApp {{
    background: {SLATE_BG};
}}

#MainMenu, footer, header {{visibility: hidden;}}
.block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1200px; }}

/* ---------- Brand header ---------- */
.brand-row {{
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 4px;
}}
.brand-tile {{
    width: 46px; height: 46px; border-radius: 12px;
    background: linear-gradient(135deg, {PRIMARY}, {PRIMARY_DARK});
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 800; font-size: 1.3rem;
    box-shadow: 0 4px 14px rgba(37,99,235,0.28);
    flex-shrink: 0;
}}
.brand-name {{
    font-size: 1.55rem; font-weight: 800; color: {TEXT_MAIN};
    letter-spacing: -0.02em; line-height: 1.1;
}}
.brand-sub {{
    color: {TEXT_MUTED}; font-size: 0.92rem; font-weight: 500; margin-top: 1px;
}}

/* ---------- Cards ---------- */
.card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}}
.section-label {{
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em;
    color: {TEXT_MUTED}; text-transform: uppercase; margin-bottom: 0.6rem;
}}

/* ---------- Risk banner ---------- */
.risk-card {{
    border-radius: 14px; padding: 1.1rem 1.4rem; margin-bottom: 1rem;
    border-left: 5px solid; display: flex; align-items: center; justify-content: space-between;
}}
.risk-title {{ font-weight: 700; font-size: 1.05rem; margin: 0; }}
.risk-value {{ font-weight: 800; font-size: 1.6rem; margin: 0; }}

/* ---------- Disease rows ---------- */
.disease-row {{
    padding: 0.65rem 0; border-bottom: 1px solid {BORDER};
}}
.disease-row:last-child {{ border-bottom: none; }}
.disease-top {{
    display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;
}}
.disease-name {{ color: {TEXT_MAIN}; font-weight: 600; font-size: 0.93rem; }}
.disease-pct {{ font-weight: 700; font-size: 0.93rem; }}
.bar-track {{
    width: 100%; height: 7px; background: #eef2f7; border-radius: 6px; overflow: hidden;
}}
.bar-fill {{ height: 100%; border-radius: 6px; }}

/* ---------- Footer ---------- */
.app-footer {{
    text-align: center; color: {TEXT_MUTED}; font-size: 0.8rem;
    margin-top: 2.2rem; padding-top: 1.2rem; border-top: 1px solid {BORDER};
}}

[data-testid="stFileUploader"] {{
    background: {CARD_BG};
    border: 1.5px dashed #cbd5e1;
    border-radius: 12px;
    padding: 0.6rem;
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL
# =========================================================
class DualHeadNet(nn.Module):
    def __init__(self, backbone="tf_efficientnetv2_s", num_diseases=10):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        feat = self.backbone.num_features
        self.binary_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, 1))
        self.disease_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, num_diseases))

    def forward(self, x):
        f = self.backbone(x)
        return self.binary_head(f).squeeze(1), self.disease_head(f)


@st.cache_resource
def load_model():
    model = DualHeadNet()
    state_dict = torch.load("best_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess(image: Image.Image):
    tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return tf(image=np.array(image.convert("RGB")))["image"].unsqueeze(0)


def predict(image: Image.Image, model: nn.Module):
    with torch.no_grad():
        x = preprocess(image)
        bin_out, dis_out = model(x)
        risk = torch.sigmoid(bin_out).item()
        probs = torch.sigmoid(dis_out).squeeze().numpy()
    return risk, probs


def prob_color(p: float) -> str:
    if p > 0.5:
        return RED
    if p > 0.25:
        return AMBER
    return PRIMARY


# =========================================================
# HEADER
# =========================================================
st.markdown(f"""
<div class="brand-row">
    <div class="brand-tile">R</div>
    <div>
        <div class="brand-name">RetinaScan AI</div>
        <div class="brand-sub">Retinal disease screening from fundus images · EfficientNetV2 · RFMiD</div>
    </div>
</div>
""", unsafe_allow_html=True)
st.write("")

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=56)
    st.markdown("### About")
    st.markdown(
        "**RetinaScan AI** analyzes **fundus (retina) photographs** to screen for "
        "**10 retinal conditions** using a deep learning model."
    )
    st.markdown("---")
    st.markdown("#### What image do I need?")
    st.markdown(
        "A **fundus photograph** — a medical image of the retina captured by an "
        "**ophthalmologist** with a fundus camera.\n\n"
        "It is **not** a regular eye selfie or phone photo."
    )
    st.markdown("---")
    st.markdown("#### Who is this for?")
    st.markdown("- Eye doctors & clinics\n- Medical students\n- Researchers")
    st.markdown("---")
    st.markdown("#### Model")
    st.markdown(
        "- Architecture: **EfficientNetV2‑S**\n"
        "- Trained on: **RFMiD** dataset\n"
        "- Detects: **10 conditions**\n"
        "- Validation score: **0.9356**"
    )
    st.markdown("---")
    st.caption("For research/demo purposes only. Not a medical diagnosis.")

# =========================================================
# LOAD MODEL
# =========================================================
model = None
model_ok = False
if not os.path.exists("best_model.pth"):
    st.error(
        "**Model weights not found.** Place `best_model.pth` in the app's root directory "
        "to enable predictions.",
        icon="⚠️",
    )
else:
    try:
        model = load_model()
        model_ok = True
    except Exception as e:
        st.error(f"Model could not be loaded: {e}", icon="⚠️")

# =========================================================
# INFO BANNER
# =========================================================
st.info(
    "This tool requires a **fundus (retina) image** taken by a medical camera — "
    "not a normal eye photo. Upload one below or try a sample.",
    icon="ℹ️",
)

# =========================================================
# MAIN LAYOUT
# =========================================================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop a fundus image here",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        st.markdown("**Or try a sample:**")
        sample_choice = st.selectbox(
            "Sample images",
            ["None", "Sample 1", "Sample 2", "Sample 3"],
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

# Resolve which image to use
image = None
if uploaded is not None:
    image = Image.open(uploaded)
elif sample_choice != "None":
    idx = {"Sample 1": "1", "Sample 2": "2", "Sample 3": "3"}[sample_choice]
    sample_path = os.path.join("samples", f"sample{idx}.png")
    if os.path.exists(sample_path):
        image = Image.open(sample_path)
    else:
        with col_left:
            st.warning("Sample image not found. Please upload your own fundus image.")

with col_left:
    if image is not None:
        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, caption="Selected image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# RESULTS
# =========================================================
with col_right:
    st.markdown('<div class="section-label">Analysis Results</div>', unsafe_allow_html=True)

    if image is not None and model_ok:
        with st.spinner("Analyzing image..."):
            risk, probs = predict(image, model)

        if risk > 0.5:
            st.markdown(f"""
            <div class="risk-card" style="background:#fef2f2; border-left-color:{RED};">
                <div>
                    <p class="risk-title" style="color:{RED};">Disease Likely Detected</p>
                    <p style="color:#991b1b; margin:0; font-size:0.85rem;">Overall screening risk</p>
                </div>
                <p class="risk-value" style="color:{RED};">{risk:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-card" style="background:#f0fdf4; border-left-color:{GREEN};">
                <div>
                    <p class="risk-title" style="color:{GREEN};">Likely Normal</p>
                    <p style="color:#166534; margin:0; font-size:0.85rem;">Overall screening risk</p>
                </div>
                <p class="risk-value" style="color:{GREEN};">{risk:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Per-Disease Probabilities</div>', unsafe_allow_html=True)

        order = np.argsort(probs)[::-1]
        rows_html = ""
        for i in order:
            code = DISEASE_ORDER[i]
            pct = float(probs[i])
            color = prob_color(pct)
            rows_html += f"""
            <div class="disease-row">
                <div class="disease-top">
                    <span class="disease-name">{DISEASE_NAMES[code]}</span>
                    <span class="disease-pct" style="color:{color};">{pct:.1%}</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct*100:.1f}%; background:{color};"></div>
                </div>
            </div>
            """
        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif image is not None and not model_ok:
        st.markdown(
            '<div class="card" style="text-align:center; color:#64748b;">'
            'Model is not loaded, so predictions are unavailable. Add <code>best_model.pth</code> '
            'to the app root and reload.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="card" style="text-align:center; color:#64748b; padding:2.4rem 1rem;">'
            'Upload a fundus image or select a sample to see the analysis.'
            '</div>',
            unsafe_allow_html=True,
        )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<p class="app-footer">RetinaScan AI · EfficientNetV2 · RFMiD · '
    'For research and educational use only. This is not a substitute for professional medical advice.</p>',
    unsafe_allow_html=True,
)