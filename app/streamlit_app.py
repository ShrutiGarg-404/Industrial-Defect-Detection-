"""
DefectLens — AI-Powered Industrial Defect Detection
Built with PatchCore anomaly detection on the MVTec AD dataset.
"""
try:
    from opencv_fixer import AutoFix
    AutoFix()
except ImportError:
    pass


import streamlit as st
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import json
import random
import time
import base64
from io import BytesIO
from datetime import datetime
from torchvision.transforms import v2 as T
import plotly.graph_objects as go


# Auto-download models if missing (for Streamlit Cloud deployment)
from pathlib import Path as _Path
_models_dir = _Path(__file__).parent.parent / "outputs" / "models"
if not all((_models_dir / f"patchcore_{c}_full.pth").exists() for c in ["leather", "tile", "metal_nut", "wood"]):
    import sys
    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from download_models import download_models
    download_models(_models_dir)

    
st.set_page_config(
    page_title="DefectLens",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
DATA_ROOT = PROJECT_ROOT / "data" / "mvtec"
RESULTS_DIR = PROJECT_ROOT / "results"

CATEGORIES = ["leather", "tile", "metal_nut", "wood"]
CATEGORY_LABELS = {"leather": "Leather", "tile": "Tile", "metal_nut": "Metal Nut", "wood": "Wood"}

COLORS = {
    "bg_deep": "#0B0D11", "bg_panel": "#14171D", "bg_raised": "#1A1E26",
    "steel": "#262B36", "text_primary": "#F2F3F5", "text_secondary": "#8B92A3",
    "amber": "#FF8A3D", "green": "#3DDC84", "red": "#FF4D6A",
}


# ============================================================
# CSS
# ============================================================
def inject_css():
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] { display: block !important; transform: none !important; }
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-deep: #0B0D11; --bg-panel: #14171D; --bg-panel-raised: #1A1E26;
        --steel: #262B36; --steel-light: #343B49;
        --text-primary: #F2F3F5; --text-secondary: #8B92A3; --text-muted: #565E6E;
        --amber: #FF8A3D; --green: #3DDC84; --red: #FF4D6A;
        --mono: 'JetBrains Mono', monospace; --display: 'Space Grotesk', sans-serif;
    }

    html, body, [class*="css"] { font-family: var(--display); }

    .stApp {
        background:
            radial-gradient(ellipse 800px 400px at 20% -10%, rgba(255,138,61,0.08), transparent),
            radial-gradient(ellipse 600px 400px at 100% 10%, rgba(61,220,132,0.04), transparent),
            var(--bg-deep);
    }
    #MainMenu, footer, header { visibility: hidden; }

    section[data-testid="stSidebar"] { background: var(--bg-panel); border-right: 1px solid var(--steel); }
    section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

    h1 { font-family: var(--display); font-weight: 700; letter-spacing: -0.03em; color: var(--text-primary) !important; font-size: 2.6rem !important; }
    h2, h3 { font-family: var(--display); color: var(--text-primary) !important; }
    p, .stMarkdown { color: var(--text-secondary); }

    .eyebrow {
        font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.22em; text-transform: uppercase;
        color: var(--amber); margin-bottom: 6px; display: flex; align-items: center; gap: 8px;
    }
    .eyebrow::before { content: ''; width: 6px; height: 6px; background: var(--amber); border-radius: 50%; box-shadow: 0 0 8px var(--amber); }

    .bay-panel {
        background: linear-gradient(180deg, var(--bg-panel-raised), var(--bg-panel));
        border: 1px solid var(--steel); border-radius: 8px; position: relative; overflow: hidden;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .bay-corner { position: absolute; width: 22px; height: 22px; border-color: var(--amber); opacity: 0.85; z-index: 5; }
    .bay-corner.tl { top: 10px; left: 10px; border-top: 2px solid; border-left: 2px solid; border-radius: 4px 0 0 0; }
    .bay-corner.tr { top: 10px; right: 10px; border-top: 2px solid; border-right: 2px solid; border-radius: 0 4px 0 0; }
    .bay-corner.bl { bottom: 10px; left: 10px; border-bottom: 2px solid; border-left: 2px solid; border-radius: 0 0 0 4px; }
    .bay-corner.br { bottom: 10px; right: 10px; border-bottom: 2px solid; border-right: 2px solid; border-radius: 0 0 4px 0; }

    .bay-label {
        font-family: var(--mono); font-size: 0.66rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--text-muted); text-align: center; padding: 10px 0 14px 0; border-top: 1px solid var(--steel);
    }
    .bay-imgwrap { position: relative; overflow: hidden; }

    @keyframes sweepDown {
        0% { top: -4%; opacity: 0; } 10% { opacity: 1; } 90% { opacity: 1; } 100% { top: 100%; opacity: 0; }
    }
    .scan-sweep {
        position: absolute; left: 0; right: 0; height: 3px; top: -4%;
        background: linear-gradient(90deg, transparent, var(--amber) 20%, #FFD9B3 50%, var(--amber) 80%, transparent);
        box-shadow: 0 0 20px 4px var(--amber), 0 0 40px 8px rgba(255,138,61,0.4);
        animation: sweepDown 1.6s cubic-bezier(0.45,0,0.55,1) 1; z-index: 20;
    }
    .scan-overlay {
        position: absolute; inset: 0;
        background: linear-gradient(180deg, rgba(255,138,61,0.0) 0%, rgba(255,138,61,0.10) 50%, rgba(255,138,61,0.0) 100%);
        animation: sweepDown 1.6s cubic-bezier(0.45,0,0.55,1) 1; z-index: 19;
    }

    @keyframes fadeUp { from { opacity:0; transform: translateY(12px);} to {opacity:1; transform: translateY(0);} }
    .verdict-hero { border-radius: 12px; overflow: hidden; position: relative; animation: fadeUp 0.5s ease-out; border: 1px solid var(--steel); }
    .verdict-top { padding: 32px 36px 24px 36px; position: relative; }
    .verdict-top.defective { background: radial-gradient(ellipse 500px 200px at 0% 0%, rgba(255,77,106,0.16), transparent), var(--bg-panel-raised); }
    .verdict-top.good { background: radial-gradient(ellipse 500px 200px at 0% 0%, rgba(61,220,132,0.16), transparent), var(--bg-panel-raised); }
    .verdict-num { font-family: var(--mono); font-size: 0.85rem; color: var(--text-muted); letter-spacing: 0.1em; }
    .verdict-title { font-family: var(--display); font-weight: 700; font-size: 3.2rem; letter-spacing: -0.03em; line-height: 1; margin: 6px 0 0 0; display: flex; align-items: baseline; gap: 16px; }
    .verdict-title.defective { color: var(--red); text-shadow: 0 0 30px rgba(255,77,106,0.35); }
    .verdict-title.good { color: var(--green); text-shadow: 0 0 30px rgba(61,220,132,0.35); }
    .verdict-sub { font-family: var(--mono); font-size: 0.95rem; color: var(--text-secondary); margin-top: 4px; }
    .verdict-bottom { background: var(--bg-panel); padding: 22px 36px 28px 36px; }

    .gauge-track { position: relative; height: 8px; background: var(--steel); border-radius: 4px; margin: 10px 0 8px 0; }
    .gauge-fill { height: 100%; border-radius: 4px; }
    .gauge-marker { position: absolute; top: -5px; width: 3px; height: 18px; background: var(--text-primary); border-radius: 2px; box-shadow: 0 0 8px rgba(255,255,255,0.6); }
    .gauge-scale { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 0.66rem; color: var(--text-muted); margin-top: 6px; }

    .truth-row {
        margin-top: 18px; padding-top: 18px; border-top: 1px solid var(--steel); font-family: var(--mono);
        font-size: 0.78rem; color: var(--text-secondary); display: flex; justify-content: space-between;
        align-items: center; flex-wrap: wrap; gap: 8px;
    }

    .chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
    .chip { font-family: var(--mono); font-size: 0.72rem; background: var(--steel); color: var(--text-secondary); padding: 5px 12px; border-radius: 20px; border: 1px solid var(--steel-light); }
    .chip.live { color: var(--green); border-color: rgba(61,220,132,0.3); background: rgba(61,220,132,0.08); }

    .stButton > button {
        background: linear-gradient(135deg, var(--amber), #FF6B35) !important; color: #1A1208 !important;
        font-family: var(--display); font-weight: 700; border: none !important; border-radius: 8px !important;
        letter-spacing: 0.02em; box-shadow: 0 4px 16px rgba(255,138,61,0.3); transition: all 0.2s; padding: 0.6rem 0 !important;
    }
    .stButton > button:hover { filter: brightness(1.08); transform: translateY(-1px); box-shadow: 0 6px 20px rgba(255,138,61,0.4); }

    .stSelectbox > div > div, .stTextInput > div > div { background: var(--bg-panel-raised) !important; border-color: var(--steel) !important; color: var(--text-primary) !important; }

    div[data-testid="stMetric"] { background: var(--bg-panel); border: 1px solid var(--steel); border-radius: 8px; padding: 16px 18px; }
    div[data-testid="stMetricValue"] { font-family: var(--mono) !important; color: var(--amber) !important; }
    div[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }

    .history-row { display: flex; align-items: center; gap: 10px; padding: 9px 12px; background: var(--bg-panel); border: 1px solid var(--steel); border-radius: 6px; margin-bottom: 5px; font-family: var(--mono); font-size: 0.76rem; }
    .history-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 6px currentColor; }

    .empty-state { padding: 80px 20px; text-align: center; border: 1px dashed var(--steel); border-radius: 12px; background: var(--bg-panel); }

    /* ===== Hero banner (About / Gallery / Pipeline) ===== */
    .page-hero {
        border-radius: 14px; padding: 48px 44px; margin-bottom: 8px; position: relative; overflow: hidden;
        background: radial-gradient(ellipse 700px 300px at 0% 0%, rgba(255,138,61,0.14), transparent), var(--bg-panel-raised);
        border: 1px solid var(--steel);
        animation: fadeUp 0.5s ease-out;
    }
    .page-hero h1 { font-size: 2.4rem !important; margin: 4px 0 10px 0 !important; }
    .page-hero p { font-size: 1.02rem; max-width: 640px; }

    /* ===== Stat strip ===== */
    .stat-strip { display: flex; gap: 14px; flex-wrap: wrap; margin: 18px 0 24px 0; }
    .stat-card {
        flex: 1; min-width: 140px; background: var(--bg-panel); border: 1px solid var(--steel); border-radius: 10px;
        padding: 16px 18px; animation: fadeUp 0.5s ease-out;
    }
    .stat-card .label { font-family: var(--mono); font-size: 0.66rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); }
    .stat-card .value { font-family: var(--mono); font-size: 1.8rem; font-weight: 700; color: var(--text-primary); margin-top: 4px; }
    .stat-card .value.amber { color: var(--amber); }
    .stat-card .value.green { color: var(--green); }
    .stat-card .value.red { color: var(--red); }

    /* ===== Pipeline steps ===== */
    .pipeline-step {
        background: var(--bg-panel); border: 1px solid var(--steel); border-radius: 10px;
        padding: 22px 24px; position: relative; height: 100%;
        transition: border-color 0.2s, transform 0.2s;
    }
    .pipeline-step:hover { border-color: var(--amber); transform: translateY(-3px); }
    .pipeline-step .step-num {
        font-family: var(--mono); font-size: 2rem; font-weight: 700; color: var(--steel-light);
        position: absolute; top: 12px; right: 16px;
    }
    .pipeline-step h4 { font-family: var(--display); color: var(--text-primary); font-size: 1.05rem; margin: 0 0 8px 0; }
    .pipeline-step p { font-size: 0.85rem; color: var(--text-secondary); margin: 0; }
    .pipeline-step .tag { display: inline-block; margin-top: 10px; font-family: var(--mono); font-size: 0.68rem; color: var(--amber); background: rgba(255,138,61,0.1); padding: 3px 9px; border-radius: 12px; }

    .pipeline-arrow { text-align: center; color: var(--steel-light); font-size: 1.4rem; padding-top: 40px; }

    /* ===== Card grid (About) ===== */
    .info-card {
        background: var(--bg-panel); border: 1px solid var(--steel); border-radius: 10px; padding: 22px 24px; height: 100%;
    }
    .info-card h4 { color: var(--text-primary); font-family: var(--display); margin: 0 0 10px 0; }
    .info-card p { font-size: 0.88rem; line-height: 1.6; margin: 0; }
    .info-card .icon { font-size: 1.6rem; margin-bottom: 10px; display: block; }

    .limitation-banner {
        background: linear-gradient(90deg, rgba(255,138,61,0.12), transparent); border-left: 3px solid var(--amber);
        border-radius: 6px; padding: 16px 20px; margin: 20px 0; font-size: 0.88rem; color: var(--text-secondary);
    }
    .limitation-banner b { color: var(--amber); }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# DATA / MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model_and_threshold(category):
    model_path = MODELS_DIR / f"patchcore_{category}_full.pth"
    if not model_path.exists():
        return None, None, None
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()
    pp = model.post_processor
    threshold = float(pp.image_threshold)
    stats = {"min": float(pp.image_min), "max": float(pp.image_max), "threshold": threshold}
    return model, threshold, stats


@st.cache_data(show_spinner=False)
def load_results_json():
    results_path = RESULTS_DIR / "patchcore_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def get_sample_test_images(category, n=8):
    test_dir = DATA_ROOT / category / "test"
    if not test_dir.exists():
        return []
    samples = []
    for subfolder in sorted(test_dir.iterdir()):
        if subfolder.is_dir():
            imgs = list(subfolder.glob("*.png"))
            for img_path in imgs[:2]:
                samples.append((str(img_path), subfolder.name))
    random.seed(42)
    random.shuffle(samples)
    return samples[:n]


@st.cache_data(show_spinner=False)
def get_dataset_counts():
    counts = {}
    for cat in CATEGORIES:
        train_dir = DATA_ROOT / cat / "train" / "good"
        test_dir = DATA_ROOT / cat / "test"
        n_train = len(list(train_dir.glob("*.png"))) if train_dir.exists() else 0
        n_test = sum(len(list(f.glob("*.png"))) for f in test_dir.iterdir() if f.is_dir()) if test_dir.exists() else 0
        counts[cat] = {"train": n_train, "test": n_test}
    return counts


def run_inference(model, threshold, pil_image):
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    img_tensor = transform(pil_image.convert("RGB")).unsqueeze(0)
    start = time.time()
    with torch.no_grad():
        output = model.model(img_tensor)
    elapsed = time.time() - start
    anomaly_map = output.anomaly_map.squeeze().cpu().numpy()
    raw_score = output.pred_score.item()
    is_defective = raw_score > threshold
    return anomaly_map, raw_score, is_defective, elapsed


def overlay_heatmap(image_pil, anomaly_map, alpha=0.45):
    image_np = np.array(image_pil.convert("RGB").resize((224, 224)))
    heatmap = anomaly_map.copy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    h, w = image_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap_colored, (w, h))
    overlay = (image_np * (1 - alpha) + heatmap_resized * alpha).astype(np.uint8)
    return image_np, heatmap_resized, overlay


def to_b64(image_array):
    img = Image.fromarray(image_array) if isinstance(image_array, np.ndarray) else image_array
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ============================================================
# UI COMPONENTS
# ============================================================

def render_bay_panel(image_array, label, animate=False, full_width=False):
    b64 = to_b64(image_array)
    sweep_html = '<div class="scan-overlay"></div><div class="scan-sweep"></div>' if animate else ""
    wrapper_open = '<div>' if full_width else '<div style="max-width:320px; margin:0 auto;">'
    html = (
        wrapper_open
        + '<div class="bay-panel">'
        + '<div class="bay-corner tl"></div><div class="bay-corner tr"></div>'
        + '<div class="bay-corner bl"></div><div class="bay-corner br"></div>'
        + '<div class="bay-imgwrap">'
        + f'<img src="data:image/png;base64,{b64}" style="width:100%; height:auto; display:block;">'
        + f'{sweep_html}'
        + '</div>'
        + f'<div class="bay-label">{label}</div>'
        + '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_verdict(is_defective, raw_score, threshold, stats, elapsed_ms, true_label=None):
    status = "DEFECTIVE" if is_defective else "PASS"
    cls = "defective" if is_defective else "good"
    span = max(stats["max"] - stats["min"], 1e-6)
    pos_pct = max(0, min(100, (raw_score - stats["min"]) / span * 100))
    thresh_pct = max(0, min(100, (threshold - stats["min"]) / span * 100))
    gradient = f"linear-gradient(90deg, var(--green), var(--amber) {thresh_pct}%, var(--red))"

    truth_html = ""
    if true_label:
        actual = "GOOD" if true_label == "good" else f"DEFECTIVE · {true_label.upper()}"
        match = (is_defective == (true_label != "good"))
        match_html = (
            '<span style="color:var(--green)">MATCHES GROUND TRUTH</span>' if match else
            '<span style="color:var(--red)">DIFFERS FROM GROUND TRUTH</span>'
        )
        truth_html = (
            '<div class="truth-row"><span>GROUND TRUTH: <b style="color:var(--text-primary)">'
            + actual + '</b></span>' + match_html + '</div>'
        )

    html = (
        '<div class="verdict-hero">'
        f'<div class="verdict-top {cls}">'
        f'<div class="verdict-num">SCAN RESULT · {elapsed_ms*1000:.0f}ms INFERENCE</div>'
        f'<div class="verdict-title {cls}">{status}</div>'
        f'<div class="verdict-sub">Anomaly score {raw_score:.2f} against calibrated threshold {threshold:.2f}</div>'
        '</div>'
        '<div class="verdict-bottom">'
        '<div class="gauge-track">'
        f'<div class="gauge-fill" style="width:100%; background:{gradient};"></div>'
        f'<div class="gauge-marker" style="left:{pos_pct}%;"></div>'
        '</div>'
        '<div class="gauge-scale">'
        f"<span>{stats['min']:.0f} · NORMAL</span><span>THRESHOLD {threshold:.1f}</span><span>ANOMALOUS · {stats['max']:.0f}</span>"
        '</div>'
        f'{truth_html}'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def stat_card(label, value, color_cls=""):
    return f"""<div class="stat-card"><div class="label">{label}</div><div class="value {color_cls}">{value}</div></div>"""


def page_hero(eyebrow, title, subtitle):
    st.markdown(f"""
    <div class="page-hero">
        <div class="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# APP START
# ============================================================
inject_css()

if "history" not in st.session_state:
    st.session_state.history = []
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()

st.sidebar.markdown('<div class="eyebrow">Industrial Inspection System</div>', unsafe_allow_html=True)
st.sidebar.title("DefectLens")
st.sidebar.markdown(
    '<p style="color:var(--text-secondary); font-size:0.88rem;">AI-powered defect detection using <b style="color:var(--text-primary)">PatchCore</b> anomaly detection.</p>',
    unsafe_allow_html=True,
)

n_inspections = len(st.session_state.history)
n_defective = sum(1 for h in st.session_state.history if h["defective"])
st.sidebar.markdown(f"""
<div class="chip-row">
    <span class="chip live">● SYSTEM ONLINE</span>
</div>
<div style="font-family:var(--mono); font-size:0.72rem; color:var(--text-muted); margin-top:10px;">
    Session inspections: <b style="color:var(--text-primary)">{n_inspections}</b><br>
    Flags raised: <b style="color:var(--red)">{n_defective}</b>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Live Inspection", "Quick Gallery", "How It Works", "About"], label_visibility="collapsed")
st.sidebar.markdown("---")

if st.session_state.history:
    st.sidebar.markdown('<div class="eyebrow">Session Log</div>', unsafe_allow_html=True)
    for entry in reversed(st.session_state.history[-6:]):
        dot_color = "var(--red)" if entry["defective"] else "var(--green)"
        st.sidebar.markdown(f"""
        <div class="history-row">
            <span class="history-dot" style="background:{dot_color}; color:{dot_color};"></span>
            <span style="color:var(--text-secondary); flex:1;">{entry['category']}</span>
            <span style="color:var(--text-primary);">{entry['score']:.1f}</span>
            <span style="color:var(--text-muted); font-size:0.68rem;">{entry['time']}</span>
        </div>
        """, unsafe_allow_html=True)
    if st.sidebar.button("Clear log", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.sidebar.markdown("---")

# st.sidebar.caption("Built during CSIR-CSIO internship · Trained on MVTec AD dataset")


# ============================================================
# PAGE: LIVE INSPECTION
# ============================================================
if page == "Live Inspection":
    st.markdown('<div class="eyebrow">Real-Time Anomaly Detection</div>', unsafe_allow_html=True)
    st.title("Live Defect Inspection")
    st.markdown("Select a material category, then upload an image or pick a sample from the test set to run inference through the trained PatchCore pipeline.")

    if st.session_state.history:
        avg_score = sum(h["score"] for h in st.session_state.history) / len(st.session_state.history)
        pass_rate = 100 * (1 - n_defective / n_inspections)
        st.markdown(f"""
        <div class="stat-strip">
            {stat_card("INSPECTIONS RUN", n_inspections, "amber")}
            {stat_card("DEFECTS FLAGGED", n_defective, "red")}
            {stat_card("PASS RATE", f"{pass_rate:.0f}%", "green")}
            {stat_card("AVG SCORE", f"{avg_score:.1f}")}
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        category = st.selectbox("Material category", CATEGORIES, format_func=lambda c: CATEGORY_LABELS[c])
        model, auto_threshold, stats = load_model_and_threshold(category)

        if model is None:
            st.error(f"Model file not found for '{category}'.")
            st.stop()

        st.markdown(f"""
        <div class="chip-row">
            <span class="chip live">● MODEL ACTIVE</span>
            <span class="chip">PatchCore · WideResNet50</span>
            <span class="chip">{CATEGORY_LABELS[category]}</span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Detection sensitivity"):
            st.caption(f"Calibrated threshold: {stats['threshold']:.1f} · Range: {stats['min']:.1f}-{stats['max']:.1f}")
            threshold = st.slider(
                "Anomaly threshold", min_value=float(stats["min"]), max_value=float(stats["max"]),
                value=float(auto_threshold), step=0.1,
                help="Images scoring above this value are flagged as defective.",
            )

        input_mode = st.radio("Input source", ["Upload image", "Test sample"], horizontal=True)
        selected_image = None
        true_label = None

        # if input_mode == "Upload image":
        #     uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        #     if uploaded_file:
        #         selected_image = Image.open(uploaded_file)
        if input_mode == "Upload image":
            uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
            if uploaded_file:
                selected_image = Image.open(uploaded_file).convert("RGB")
        else:
            samples = get_sample_test_images(category, n=8)
            if samples:
                options = [f"{Path(p).name} · {label}" for p, label in samples]
                choice = st.selectbox("Sample image", options, label_visibility="collapsed")
                idx = options.index(choice)
                img_path, true_label = samples[idx]
                selected_image = Image.open(img_path)

        run_button = st.button("RUN INSPECTION", type="primary", use_container_width=True)

    with col2:
        if selected_image is not None and run_button:
            with st.spinner(""):
                anomaly_map, pred_score, is_defective, elapsed = run_inference(model, threshold, selected_image)
                original, heatmap, overlay = overlay_heatmap(selected_image, anomaly_map)

            st.session_state.history.append({
                "category": CATEGORY_LABELS[category], "score": pred_score,
                "defective": is_defective, "time": datetime.now().strftime("%H:%M:%S"),
            })

            img_col1, img_col2, img_col3 = st.columns(3)
            # 
            with img_col1:
                render_bay_panel(original, "ORIGINAL", animate=True, full_width=True)
            with img_col2:
                render_bay_panel(heatmap, "ANOMALY MAP", animate=True, full_width=True)
            with img_col3:
                render_bay_panel(overlay, "OVERLAY", animate=True, full_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            render_verdict(is_defective, pred_score, threshold, stats, elapsed, true_label)


        elif selected_image is not None:
            preview_arr = np.array(selected_image.convert("RGB").resize((280, 280)))
            render_bay_panel(preview_arr, "READY - AWAITING INSPECTION")
        else:
            st.markdown("""
            <div class="empty-state">
                <div style="font-family:var(--mono); color:var(--text-muted); font-size:0.85rem; letter-spacing:0.08em;">
                    NO IMAGE SELECTED<br><br>Select a category and image to begin inspection
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# PAGE: QUICK GALLERY
# ============================================================
elif page == "Quick Gallery":
    page_hero(
        "Pre-Computed Benchmarks",
        "Results Gallery",
        "Performance metrics and defect localization examples computed during formal model evaluation on the MVTec AD test set.",
    )

    results = load_results_json()
    counts = get_dataset_counts()

    if results:
        avg_auroc = sum(m["image_AUROC"] for m in results.values()) / len(results)
        total_test = sum(c["test"] for c in counts.values())
        st.markdown(f"""
        <div class="stat-strip">
            {stat_card("AVG IMAGE AUROC", f"{avg_auroc:.3f}", "amber")}
            {stat_card("CATEGORIES", len(results), "green")}
            {stat_card("TEST IMAGES", total_test)}
            {stat_card("BACKBONE", "WideRN50")}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Per-category performance")
        fig = go.Figure()
        cats = list(results.keys())
        img_aurocs = [results[c]["image_AUROC"] for c in cats]
        pix_aurocs = [results[c]["pixel_AUROC"] for c in cats]

        fig.add_trace(go.Bar(
            x=[CATEGORY_LABELS.get(c, c) for c in cats], y=img_aurocs, name="Image AUROC",
            marker_color=COLORS["amber"], text=[f"{v:.3f}" for v in img_aurocs], textposition="outside",
        ))
        fig.add_trace(go.Bar(
            x=[CATEGORY_LABELS.get(c, c) for c in cats], y=pix_aurocs, name="Pixel AUROC",
            marker_color=COLORS["green"], text=[f"{v:.3f}" for v in pix_aurocs], textposition="outside",
        ))
        fig.update_layout(
            barmode="group", plot_bgcolor=COLORS["bg_panel"], paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono", color=COLORS["text_secondary"], size=12),
            yaxis=dict(range=[0, 1.15], gridcolor=COLORS["steel"], title="AUROC"),
            xaxis=dict(gridcolor=COLORS["steel"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=30, b=10, l=10, r=10), height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Defect Localization Examples")
    tab1, tab2 = st.tabs(["Leather · Texture", "Metal Nut · Object"])
    with tab1:
        img_path = RESULTS_DIR / "patchcore_heatmaps_leather.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
    with tab2:
        img_path = RESULTS_DIR / "patchcore_heatmaps_metal_nut.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)

    st.markdown("---")
    st.subheader("Baseline Comparison · ResNet18 vs PatchCore")
    chart_path = RESULTS_DIR / "resnet_vs_patchcore_comparison.png"
    if chart_path.exists():
        st.image(str(chart_path), use_container_width=True)


# ============================================================
# PAGE: HOW IT WORKS
# ============================================================
elif page == "How It Works":
    page_hero(
        "System Architecture",
        "How DefectLens Works",
        "An end-to-end anomaly detection pipeline — from raw image to localized defect verdict — built without requiring a single labeled defective training example.",
    )

    steps = [
        ("01", "Image Input", "Input image is resized to 224×224 and normalized to match the backbone's expected distribution.", "224 × 224 px"),
        ("02", "Feature Extraction", "A WideResNet50 pretrained on ImageNet extracts mid-level features from layer2 and layer3 — capturing texture and structure without needing fine-tuning.", "24.9M params"),
        ("03", "Memory Bank Lookup", "Each image patch is compared against a coreset memory bank built entirely from defect-free training images — 10% of all patches retained via greedy coreset sampling.", "~25K patches"),
        ("04", "Anomaly Scoring", "Patches that sit far from anything in the memory bank receive high anomaly scores, building a per-pixel anomaly map across the image.", "Per-patch distance"),
        ("05", "Calibrated Verdict", "The image-level score is compared against a threshold fitted during evaluation, converting the raw anomaly map into a calibrated GOOD / DEFECTIVE decision.", "Auto-calibrated"),
    ]

    cols = st.columns(len(steps))
    for i, (num, title, desc, tag) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-num">{num}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
                <div class="tag">{tag}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Why PatchCore over a standard classifier")
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown("""
        <div class="info-card">
            <span class="icon">⚠</span>
            <h4>The problem with classification</h4>
            <p>An early baseline using ResNet18 as a binary classifier was trained on this dataset, which contains
            <b style="color:var(--text-primary)">only defect-free training images</b>. With nothing to contrast against,
            the classifier collapsed to always predicting "good" — a documented limitation that motivated the
            switch to anomaly detection.</p>
        </div>
        """, unsafe_allow_html=True)
    with info_col2:
        st.markdown("""
        <div class="info-card">
            <span class="icon">✓</span>
            <h4>Why PatchCore fits</h4>
            <p>PatchCore never needs to see a defect during training — it only needs to know what "normal" looks like.
            This mirrors real industrial settings, where defective samples are rare and expensive to label, but
            defect-free production output is abundant.</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE: ABOUT
# ============================================================
else:
    page_hero(
        "Project Overview",
        "About DefectLens",
        "An AI-powered visual inspection system for automated defect detection in industrial components "
        # "built during a research internship at CSIR-CSIO, Chandigarh.",
    )

    results = load_results_json()
    counts = get_dataset_counts()
    total_train = sum(c["train"] for c in counts.values())
    total_test = sum(c["test"] for c in counts.values())

    st.markdown(f"""
    <div class="stat-strip">
        {stat_card("CATEGORIES", len(CATEGORIES), "amber")}
        {stat_card("TRAINING IMAGES", total_train, "green")}
        {stat_card("TEST IMAGES", total_test)}
        {stat_card("AVG AUROC", f"{(sum(m['image_AUROC'] for m in results.values())/len(results)):.3f}" if results else "—", "amber")}
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="info-card">
            <span class="icon">◆</span>
            <h4>Dataset</h4>
            <p>Trained and evaluated on the MVTec Anomaly Detection dataset across four categories:
            leather, tile, metal_nut, and wood — covering both texture-based and object-based defect types.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-card">
            <span class="icon">◆</span>
            <h4>Model</h4>
            <p>PatchCore anomaly detection with a WideResNet50 backbone, achieving 98%+ image-level AUROC
            across all categories without ever training on a single defective example.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="info-card">
            <span class="icon">◆</span>
            <h4>Deployment</h4>
            <p>This interactive Streamlit application — live inference, defect localization heatmaps, and
            a session-based inspection log, all running on the trained models in real time.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="limitation-banner">
        <b>Known limitation:</b> The metal_nut category shows reduced precision on borderline "good" samples in
        this live demo due to threshold sensitivity near the decision boundary. The underlying model still achieves
        99.46% image-level AUROC during formal evaluation — true defect detection (true positives) remains accurate
        across all test cases.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Pipeline summary")
    st.markdown("""
    1. **Baseline** — ResNet18 transfer learning classifier (documented limitation with one-class data)
    2. **PatchCore** — Anomaly detection achieving 98%+ image-level AUROC across all categories
    3. **Localization** — Heatmaps pinpointing defect regions at the pixel level
    4. **Calibration** — Per-category threshold from anomalib's fitted post-processor
    5. **Deployment** — This interactive application, with live inference and session analytics
    """)

    st.markdown("---")
    st.caption("Built with PyTorch, Anomalib, Streamlit, and Plotly.")