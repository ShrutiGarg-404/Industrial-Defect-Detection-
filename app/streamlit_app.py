"""
DefectLens — AI-Powered Industrial Defect Detection
Built with PatchCore anomaly detection on the MVTec AD dataset.
"""

import streamlit as st
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import json
import random
from torchvision.transforms import v2 as T

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="DefectLens",
    page_icon="🔍",
    layout="wide",
)

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
DATA_ROOT = PROJECT_ROOT / "data" / "mvtec"
RESULTS_DIR = PROJECT_ROOT / "results"

CATEGORIES = ["leather", "tile", "metal_nut", "wood"]


# ============================================================
# CACHED MODEL + THRESHOLD LOADING
# ============================================================
# @st.cache_resource(show_spinner=False)
# def load_model_and_threshold(category):
#     """Load a trained PatchCore model and compute a calibrated threshold
#     from a larger sample of known-good training images."""
#     model_path = MODELS_DIR / f"patchcore_{category}_full.pth"
#     if not model_path.exists():
#         return None, None, None

#     model = torch.load(model_path, map_location="cpu", weights_only=False)
#     model.eval()

#     transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
#     train_dir = DATA_ROOT / category / "train" / "good"
#     train_imgs = list(train_dir.glob("*.png"))

#     sample_imgs = train_imgs[:60] if len(train_imgs) >= 60 else train_imgs

#     scores = []
#     with torch.no_grad():
#         for img_path in sample_imgs:
#             img = Image.open(img_path).convert("RGB")
#             tensor = transform(img).unsqueeze(0)
#             output = model.model(tensor)
#             scores.append(output.pred_score.item())

#     scores = np.array(scores)
#     mean_score = scores.mean()
#     std_score = scores.std()

#     threshold = mean_score + 5 * std_score

#     stats = {
#         "mean": float(mean_score),
#         "std": float(std_score),
#         "min": float(scores.min()),
#         "max": float(scores.max()),
#         "n_samples": len(sample_imgs),
#     }

#     return model, float(threshold), stats





@st.cache_resource(show_spinner=False)
def load_model_and_threshold(category):
    """Load a trained PatchCore model and use anomalib's own fitted threshold."""
    model_path = MODELS_DIR / f"patchcore_{category}_full.pth"
    if not model_path.exists():
        return None, None, None

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    pp = model.post_processor
    threshold = float(pp.image_threshold)
    stats = {
        "min": float(pp.image_min),
        "max": float(pp.image_max),
        "threshold": threshold,
    }

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


# ============================================================
# INFERENCE
# ============================================================
def run_inference(model, threshold, pil_image):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    img_tensor = transform(pil_image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        output = model.model(img_tensor)

    anomaly_map = output.anomaly_map.squeeze().cpu().numpy()
    raw_score = output.pred_score.item()
    is_defective = raw_score > threshold

    return anomaly_map, raw_score, is_defective


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


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🔍 DefectLens")
st.sidebar.markdown("AI-powered industrial defect detection using **PatchCore** anomaly detection.")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", ["Live Inspection", "Quick Gallery", "About"])

st.sidebar.markdown("---")
st.sidebar.caption("Built during CSIR-CSIO internship · Trained on MVTec AD dataset")


# ============================================================
# PAGE: LIVE INSPECTION
# ============================================================
if page == "Live Inspection":
    st.title("Live Defect Inspection")
    st.markdown("Select a material category, then upload an image or pick a sample from the test set to run **real-time PatchCore inference**.")

    col1, col2 = st.columns([1, 2])

    with col1:
        category = st.selectbox("Material category", CATEGORIES)
        model, auto_threshold, stats = load_model_and_threshold(category)

        if model is None:
            st.error(f"Model file not found for '{category}'. Check outputs/models/patchcore_{category}_full.pth")
            st.stop()

        st.success(f"Model loaded: PatchCore + WideResNet50 ({category})")

        # with st.expander("⚙️ Detection sensitivity (advanced)"):
        #     # st.caption(
        #     #     f"Calibrated from {stats['n_samples']} known-good training images. "
        #     #     f"Normal score range: {stats['min']:.1f}–{stats['max']:.1f}"
        #     # )
        #     st.caption(
        #         f"Mean: {stats['mean']:.1f} | Std: {stats['std']:.1f} | "
        #         f"Auto threshold (5σ): {auto_threshold:.1f}"
        #     )
        #     threshold = st.slider(
        #         "Anomaly threshold",
        #         min_value=float(stats["mean"]),
        #         max_value=float(stats["mean"] + 10 * stats["std"]),
        #         value=float(auto_threshold),
        #         step=0.1,
        #         help="Images scoring above this value are flagged as defective. Lower = more sensitive.",
        #     )




        with st.expander("⚙️ Detection sensitivity (advanced)"):
            st.caption(
                f"Calibrated threshold from training: {stats['threshold']:.1f} "
                f"(score range observed: {stats['min']:.1f}–{stats['max']:.1f})"
            )
            threshold = st.slider(
                "Anomaly threshold",
                min_value=float(stats["min"]),
                max_value=float(stats["max"]),
                value=float(auto_threshold),
                step=0.1,
                help="Images scoring above this value are flagged as defective. Lower = more sensitive.",
            )
        input_mode = st.radio("Choose input", ["Upload your own image", "Pick a test sample"])

        selected_image = None
        true_label = None

        if input_mode == "Upload your own image":
            uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
            if uploaded_file:
                selected_image = Image.open(uploaded_file)
        else:
            samples = get_sample_test_images(category, n=8)
            if samples:
                options = [f"{Path(p).name} ({label})" for p, label in samples]
                choice = st.selectbox("Pick a sample image", options)
                idx = options.index(choice)
                img_path, true_label = samples[idx]
                selected_image = Image.open(img_path)
            else:
                st.warning("No sample images found for this category.")

        run_button = st.button("Run Inspection", type="primary", use_container_width=True)

    with col2:
        if selected_image is not None and run_button:
            with st.spinner("Running PatchCore inference..."):
                anomaly_map, pred_score, is_defective = run_inference(model, threshold, selected_image)
                original, heatmap, overlay = overlay_heatmap(selected_image, anomaly_map)

            result_col1, result_col2, result_col3 = st.columns(3)
            with result_col1:
                st.image(original, caption="Original", use_container_width=True)
            with result_col2:
                st.image(heatmap, caption="Anomaly Heatmap", use_container_width=True)
            with result_col3:
                st.image(overlay, caption="Overlay", use_container_width=True)

            st.markdown("---")
            verdict = "🔴 DEFECTIVE" if is_defective else "🟢 GOOD"
            st.metric("Verdict", verdict, f"Score: {pred_score:.1f}  (threshold: {threshold:.1f})")

            if true_label:
                actual = "GOOD" if true_label == "good" else f"DEFECTIVE ({true_label})"
                match = "✅ Matches ground truth" if (is_defective == (true_label != "good")) else "⚠️ Differs from ground truth"
                st.caption(f"Ground truth label: **{actual}** — {match}")

        elif selected_image is not None:
            st.image(selected_image, caption="Selected image — click 'Run Inspection'", use_container_width=True)
        else:
            st.info("👈 Select a category and image to begin inspection.")


# ============================================================
# PAGE: QUICK GALLERY
# ============================================================
elif page == "Quick Gallery":
    st.title("Quick Results Gallery")
    st.markdown("Pre-computed PatchCore results across all categories — instant viewing, no inference needed.")

    results = load_results_json()

    if results:
        st.subheader("Performance Summary")
        cols = st.columns(len(results))
        for i, (cat, metrics) in enumerate(results.items()):
            with cols[i]:
                st.metric(cat.replace("_", " ").title(), f"{metrics['image_AUROC']:.3f}", "Image AUROC")

    st.markdown("---")
    st.subheader("Defect Localization Examples")

    tab1, tab2 = st.tabs(["Leather (Texture)", "Metal Nut (Object)"])
    with tab1:
        img_path = RESULTS_DIR / "patchcore_heatmaps_leather.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.warning("Heatmap image not found.")
    with tab2:
        img_path = RESULTS_DIR / "patchcore_heatmaps_metal_nut.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.warning("Heatmap image not found.")

    st.markdown("---")
    st.subheader("ResNet18 Baseline vs PatchCore")
    chart_path = RESULTS_DIR / "resnet_vs_patchcore_comparison.png"
    if chart_path.exists():
        st.image(str(chart_path), use_container_width=True)


# ============================================================
# PAGE: ABOUT
# ============================================================
else:
    st.title("About DefectLens")
    st.markdown("""
    **DefectLens** is an AI-powered visual inspection system for automated defect detection
    in industrial components, built during a research internship at **CSIR-CSIO, Chandigarh**.

    ### How it works
    The system uses **PatchCore**, a state-of-the-art anomaly detection algorithm, combined with
    a pre-trained WideResNet50 backbone. Instead of requiring labeled defective examples,
    PatchCore learns what "normal" looks like from defect-free training images, then flags
    anything that deviates — making it ideal for real-world industrial settings where defective
    samples are rare.

    ### Dataset
    Trained and evaluated on the MVTec Anomaly Detection dataset, across four categories:
    leather, tile, metal_nut, and wood.

    ### Pipeline
    1. **Baseline**: ResNet18 transfer learning classifier (documented limitation with one-class data)
    2. **PatchCore**: Anomaly detection achieving 98%+ image-level AUROC across all categories
    3. **Localization**: Heatmaps pinpointing defect regions
    4. **Calibration**: Per-category decision threshold computed from training data statistics
    5. **Deployment**: This interactive Streamlit application

    ---
    Built with PyTorch, Anomalib, and Streamlit.
    """)