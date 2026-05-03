"""
app.py — Advanced ML Image Editor
Features: Blur, Sharpness, Brightness, Contrast, Edge Detection, Grayscale,
          Noise Reduction, Color Filters, Crop, Rotate, Face Detection,
          Super-Resolution Quality Enhance
"""

import streamlit as st
from PIL import Image
import numpy as np

from filters import apply_all_filters, enhance_quality
from utils import pil_to_numpy, numpy_to_pil, numpy_to_bytes, image_info

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Image Editor Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #16192b 100%);
        border-right: 1px solid #2d3561;
    }

    /* Section headers in sidebar */
    .filter-section {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 12px 14px;
        margin: 8px 0;
    }
    .filter-section-title {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #a5b4fc;
        margin-bottom: 8px;
    }

    /* Enhance button */
    .enhance-btn > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        padding: 14px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
        transition: all 0.2s !important;
        width: 100% !important;
    }

    /* Image containers */
    .img-card {
        background: #1a1d2e;
        border: 1px solid #2d3561;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }
    .img-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #6366f1;
        margin-bottom: 10px;
    }

    /* Stat cards */
    .stat-row { display: flex; gap: 10px; margin: 10px 0; }
    .stat-card {
        flex: 1;
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 10px;
        padding: 10px;
        text-align: center;
    }
    .stat-val { font-size: 18px; font-weight: 700; color: #a5b4fc; }
    .stat-lbl { font-size: 11px; color: #6b7280; margin-top: 2px; }

    /* Color filter pills */
    .pill-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 6px 0; }
    .pill {
        padding: 4px 12px; border-radius: 20px; font-size: 12px;
        font-weight: 600; cursor: pointer;
    }

    /* Download button */
    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        width: 100% !important;
    }

    /* Metric overrides */
    [data-testid="stMetric"] {
        background: rgba(99,102,241,0.06);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 10px;
        padding: 10px !important;
    }
    [data-testid="stMetricValue"] { color: #a5b4fc !important; font-size: 16px !important; }
    [data-testid="stMetricLabel"] { color: #6b7280 !important; }

    /* Hide default streamlit footer */
    footer { visibility: hidden; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1d2e;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] { color: #6b7280 !important; }
    .stTabs [aria-selected="true"] { color: #a5b4fc !important; }
</style>
""", unsafe_allow_html=True)

# ── Session defaults ──────────────────────────────────────────────────────────
DEFAULTS = {
    "blur_ksize": 1,
    "sharpness_alpha": 1.0,
    "brightness_beta": 0,
    "contrast_alpha": 1.0,
    "edge_enabled": False,
    "edge_thresh1": 100,
    "edge_thresh2": 200,
    "grayscale_enabled": False,
    "denoise_strength": 0,
    "color_filter": "none",
    "rotate_angle": 0,
    "crop_x1": 0.0,
    "crop_y1": 0.0,
    "crop_x2": 100.0,
    "crop_y2": 100.0,
    "face_detect": False,
    "enhance_scale": 2,
    "enhanced_img": None,
    "show_enhanced": False,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_filters():
    for k, v in DEFAULTS.items():
        if k not in ("enhanced_img", "show_enhanced"):
            st.session_state[k] = v
    st.session_state["show_enhanced"] = False
    st.session_state["enhanced_img"] = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎨 ML Image Editor Pro")
    st.markdown("---")

    # ── TRANSFORM ──────────────────────────────────────────────────────
    st.markdown('<div class="filter-section"><div class="filter-section-title">✂️ Transform</div>', unsafe_allow_html=True)
    st.slider("Rotate (degrees)", -180, 180, step=1, key="rotate_angle")
    st.markdown("**Crop** (% from edges)")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.number_input("Left %", 0.0, 49.0, step=1.0, key="crop_x1", label_visibility="visible")
        st.number_input("Top %", 0.0, 49.0, step=1.0, key="crop_y1")
    with cc2:
        st.number_input("Right %", 51.0, 100.0, step=1.0, key="crop_x2")
        st.number_input("Bottom %", 51.0, 100.0, step=1.0, key="crop_y2")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── COLOUR FILTERS ──────────────────────────────────────────────────
    st.markdown('<div class="filter-section"><div class="filter-section-title">🎭 Color Style</div>', unsafe_allow_html=True)
    st.selectbox(
        "Apply color filter",
        options=["none", "warm", "cool", "sepia", "invert"],
        format_func=lambda x: {"none": "None", "warm": "🔥 Warm", "cool": "❄️ Cool",
                                "sepia": "🟤 Sepia", "invert": "🔀 Invert"}[x],
        key="color_filter",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── ADJUSTMENTS ─────────────────────────────────────────────────────
    st.markdown('<div class="filter-section"><div class="filter-section-title">🎚️ Adjustments</div>', unsafe_allow_html=True)
    st.slider("Blur", 1, 51, step=2, key="blur_ksize", help="Gaussian kernel size (odd)")
    st.slider("Sharpness", 0.0, 3.0, step=0.1, key="sharpness_alpha")
    st.slider("Brightness", -100, 100, step=1, key="brightness_beta")
    st.slider("Contrast", 0.5, 3.0, step=0.1, key="contrast_alpha")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── EFFECTS ─────────────────────────────────────────────────────────
    st.markdown('<div class="filter-section"><div class="filter-section-title">✨ Effects</div>', unsafe_allow_html=True)
    st.slider("Noise Reduction", 0, 20, step=1, key="denoise_strength",
              help="Non-local means denoising strength")
    st.checkbox("🔍 Grayscale", key="grayscale_enabled")
    st.checkbox("⚡ Edge Detection", key="edge_enabled")
    if st.session_state["edge_enabled"]:
        st.slider("Edge Threshold 1", 0, 255, key="edge_thresh1")
        st.slider("Edge Threshold 2", 0, 255, key="edge_thresh2")
    st.checkbox("👤 Face Detection", key="face_detect",
                help="Haar cascade — works best on clear frontal faces")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.button("🔄 Reset All Filters", on_click=reset_filters, use_container_width=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## 🎨 ML Image Editor")

uploaded_file = st.file_uploader(
    "Drop an image here or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    original_bgr = pil_to_numpy(pil_img)

    # ── Apply all live filters ────────────────────────────────────────
    processed_bgr = apply_all_filters(
        original=original_bgr,
        blur_ksize=st.session_state["blur_ksize"],
        sharpness_alpha=st.session_state["sharpness_alpha"],
        brightness_beta=st.session_state["brightness_beta"],
        contrast_alpha=st.session_state["contrast_alpha"],
        edge_enabled=st.session_state["edge_enabled"],
        edge_thresh1=st.session_state["edge_thresh1"],
        edge_thresh2=st.session_state["edge_thresh2"],
        grayscale_enabled=st.session_state["grayscale_enabled"],
        denoise_strength=st.session_state["denoise_strength"],
        color_filter=st.session_state["color_filter"],
        rotate_angle=st.session_state["rotate_angle"],
        crop_x1=st.session_state["crop_x1"],
        crop_y1=st.session_state["crop_y1"],
        crop_x2=st.session_state["crop_x2"],
        crop_y2=st.session_state["crop_y2"],
        face_detect=st.session_state["face_detect"],
    )
    processed_pil = numpy_to_pil(processed_bgr)

    # ── Tabs: Editor / Enhance / Info ─────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🖼️ Editor", "⚡ Enhance Quality", "📊 Image Info"])

    with tab1:
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown('<div class="img-card"><div class="img-label">Original</div>', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)
            h, w = original_bgr.shape[:2]
            st.caption(f"📐 {w} × {h} px")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="img-card"><div class="img-label">Processed</div>', unsafe_allow_html=True)
            st.image(processed_pil, use_container_width=True)
            ph, pw = processed_bgr.shape[:2]
            st.caption(f"📐 {pw} × {ph} px")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Active filters summary
        active = []
        if st.session_state["blur_ksize"] > 1: active.append(f"Blur ({st.session_state['blur_ksize']}px)")
        if abs(st.session_state["sharpness_alpha"] - 1.0) > 0.05: active.append(f"Sharpness ({st.session_state['sharpness_alpha']:.1f}×)")
        if st.session_state["brightness_beta"] != 0: active.append(f"Brightness ({st.session_state['brightness_beta']:+d})")
        if abs(st.session_state["contrast_alpha"] - 1.0) > 0.05: active.append(f"Contrast ({st.session_state['contrast_alpha']:.1f}×)")
        if st.session_state["color_filter"] != "none": active.append(f"Color: {st.session_state['color_filter'].title()}")
        if st.session_state["denoise_strength"] > 0: active.append(f"Denoise ({st.session_state['denoise_strength']})")
        if st.session_state["rotate_angle"] != 0: active.append(f"Rotate {st.session_state['rotate_angle']}°")
        if st.session_state["grayscale_enabled"]: active.append("Grayscale")
        if st.session_state["edge_enabled"]: active.append("Edges")
        if st.session_state["face_detect"]: active.append("Face Detection")

        if active:
            st.markdown(f"**Active filters:** " + " · ".join([f"`{a}`" for a in active]))
        else:
            st.markdown("**No filters active** — adjust sliders in the sidebar")

        st.download_button(
            "⬇️ Download Processed Image (PNG)",
            data=numpy_to_bytes(processed_bgr),
            file_name="processed_image.png",
            mime="image/png",
            use_container_width=True,
        )

    with tab2:
        st.markdown("### ⚡ Super-Resolution Quality Enhancer")
        st.markdown(
            "Upscales your processed image using **bicubic interpolation** + "
            "**unsharp mask sharpening** + **CLAHE contrast** + **light denoising**."
        )

        scale_col, btn_col = st.columns([1, 2])
        with scale_col:
            scale = st.selectbox("Upscale factor", [2, 3, 4],
                                  format_func=lambda x: f"{x}× ({original_bgr.shape[1]*x}×{original_bgr.shape[0]*x} px)",
                                  key="enhance_scale")
        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Enhance Quality Now", use_container_width=True, type="primary"):
                with st.spinner("Enhancing image — this may take a few seconds..."):
                    enhanced = enhance_quality(processed_bgr, scale=scale)
                    st.session_state["enhanced_img"] = enhanced
                    st.session_state["show_enhanced"] = True

        if st.session_state["show_enhanced"] and st.session_state["enhanced_img"] is not None:
            enhanced = st.session_state["enhanced_img"]
            eh, ew = enhanced.shape[:2]
            oh, ow = processed_bgr.shape[:2]

            st.success(f"✅ Enhanced: {ow}×{oh} → **{ew}×{eh} px** ({scale}× upscale)")

            ec1, ec2 = st.columns(2, gap="medium")
            with ec1:
                st.markdown("**Before**")
                st.image(processed_pil, use_container_width=True)
                st.caption(f"{ow} × {oh} px")
            with ec2:
                st.markdown("**Enhanced ✨**")
                st.image(numpy_to_pil(enhanced), use_container_width=True)
                st.caption(f"{ew} × {eh} px")

            st.download_button(
                f"⬇️ Download Enhanced Image ({ew}×{eh} PNG)",
                data=numpy_to_bytes(enhanced),
                file_name=f"enhanced_{scale}x.png",
                mime="image/png",
                use_container_width=True,
            )

    with tab3:
        st.markdown("### 📊 Image Information")
        info = image_info(original_bgr)
        pinfo = image_info(processed_bgr)

        st.markdown("**Original**")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Width", f"{info['width']} px")
        m2.metric("Height", f"{info['height']} px")
        m3.metric("Channels", info['ndim'])
        m4.metric("Mode", info['mode'].split()[0])
        m5.metric("Size", f"{info['size_kb']} KB")

        st.markdown("**Processed**")
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Width", f"{pinfo['width']} px")
        p2.metric("Height", f"{pinfo['height']} px")
        p3.metric("Channels", pinfo['ndim'])
        p4.metric("Mode", pinfo['mode'].split()[0])
        p5.metric("Size", f"{pinfo['size_kb']} KB")

        st.markdown("---")
        st.markdown("**Notebook reference: `img.mode`, `img.shape`, `img.ndim`**")
        st.code(f"""
# From ml_image_preprocessing_class1.ipynb
img = Image.open(uploaded_file)
pil_img_ary = np.array(img)

print(img.mode)       # → {info['mode'].split()[0]}
print(pil_img_ary.shape)  # → {info['shape']}
print(pil_img_ary.ndim)   # → {info['ndim']}

# Grayscale: L = R*299/1000 + G*587/1000 + B*114/1000
gry_scl_img = img.convert('L')
        """, language="python")

else:
    # ── Landing state ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **🎚️ Adjustments**
        Blur · Sharpness · Brightness · Contrast
        """)
    with c2:
        st.markdown("""
        **✨ Effects**
        Color filters · Denoise · Grayscale · Edges · Face Detection
        """)
    with c3:
        st.markdown("""
        **⚡ Enhance**
        Super-resolution upscale up to 4×
        Bicubic · CLAHE · Unsharp mask
        """)
    st.info("👆 Upload a JPG or PNG to get started")