"""
Streamlit demo — the user-facing deliverable.

Upload a traffic sign image, optionally attack it live, then see:
  * predicted class + confidence
  * clean / adversarial verdict + confidence
  * Grad-CAM heatmaps for both the classifier and the detector

    streamlit run app/streamlit_app.py
"""
import io
import os
import sys

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                    # noqa: E402
from src import models, data, generate_attacks   # noqa: E402

st.set_page_config(page_title="Adversarial Traffic Sign Detector",
                   page_icon="🛑", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_resource
def load_models():
    device = config.DEVICE
    clf = models.load_classifier(device=device)
    det = models.load_detector(device=device)
    return clf, det, device


def preprocess(pil_img):
    """PIL -> (1,3,32,32) tensor in [0,1]."""
    img = pil_img.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def to_display(tensor):
    """(1,3,H,W) in [0,1] -> uint8 HWC numpy, upscaled for viewing."""
    arr = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    img = Image.fromarray((arr * 255).astype(np.uint8))
    return img.resize((256, 256), Image.NEAREST)


def overlay_cam(tensor, cam):
    """Blend a Grad-CAM heatmap over the image."""
    import matplotlib.cm as cm
    base = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    heat = cm.jet(cam)[..., :3]
    blended = 0.55 * base + 0.45 * heat
    img = Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))
    return img.resize((256, 256), Image.BILINEAR)


# ------------------------------------------------------------------ UI
st.title("🛑 Adversarial Attack Detection for Traffic Sign Recognition")
st.caption("CNN classifier + adversarial detector + Grad-CAM explainability")

try:
    clf, det, device = load_models()
except FileNotFoundError:
    st.error(
        "Model checkpoints not found. Train them first:\n\n"
        "```\npython -m src.train_classifier\n"
        "python -m src.generate_attacks\n"
        "python -m src.train_detector\n```"
    )
    st.stop()

classes = data.class_names()

with st.sidebar:
    st.header("Attack settings")
    attack_name = st.selectbox(
        "Apply an attack to the uploaded image",
        ["none", "fgsm", "pgd", "cw", "square"],
    )
    eps_255 = st.slider("Perturbation budget ε (×1/255)", 1, 16, 8)
    st.markdown("---")
    st.caption(f"Device: `{device}`")
    st.caption("ε = 8/255 is the standard benchmark budget.")

uploaded = st.file_uploader(
    "Upload a traffic sign image", type=["png", "jpg", "jpeg", "ppm"]
)

if uploaded is None:
    st.info("Upload an image to begin. Try a GTSRB test image.")
    st.stop()

pil = Image.open(io.BytesIO(uploaded.read()))
x = preprocess(pil).to(device)

# ---- optionally attack the image live
x_input = x
if attack_name != "none":
    with torch.no_grad():
        y_pred = clf(x).argmax(dim=1)
    config.EPS = eps_255 / 255.0          # honour the slider
    with st.spinner(f"Running {attack_name.upper()} attack..."):
        atk = generate_attacks.build_attack(attack_name, clf)
        x_input = atk(x, y_pred)

# ---- classifier
with torch.no_grad():
    logits = clf(x_input)
    probs = F.softmax(logits, dim=1)
    pred_idx = int(probs.argmax())
    pred_conf = float(probs[0, pred_idx])

# ---- detector
with torch.no_grad():
    d_logits = det(x_input)
    d_probs = F.softmax(d_logits, dim=1)
    is_adv = int(d_probs.argmax())
    adv_conf = float(d_probs[0, 1])

# ---- results
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Input")
    st.image(to_display(x_input), use_container_width=False)
    if attack_name != "none":
        pert = (x_input - x).abs().max().item()
        st.caption(f"{attack_name.upper()} applied — L∞ = {pert:.4f}")

with c2:
    st.subheader("Classification")
    st.metric("Predicted sign", classes[pred_idx])
    st.metric("Confidence", f"{pred_conf:.1%}")

with c3:
    st.subheader("Detection verdict")
    if is_adv:
        st.error("⚠️ ADVERSARIAL — reject this input")
    else:
        st.success("✅ CLEAN — safe to use")
    st.metric("P(adversarial)", f"{adv_conf:.1%}")

st.markdown("---")
st.subheader("Grad-CAM — what each model looked at")

cam_clf = models.GradCAM(clf)
cam_det = models.GradCAM(det)
h_clf, _ = cam_clf(x_input.clone())
h_det, _ = cam_det(x_input.clone())

g1, g2 = st.columns(2)
with g1:
    st.image(overlay_cam(x_input, h_clf),
             caption="Classifier attention — regions driving the class prediction")
with g2:
    st.image(overlay_cam(x_input, h_det),
             caption="Detector attention — regions flagged as tampered")

with st.expander("How to read this"):
    st.write(
        "The classifier heatmap shows which pixels pushed it toward its chosen "
        "sign class. The detector heatmap shows where it found perturbation "
        "artefacts. On a clean image the detector map is usually diffuse; on an "
        "attacked image it often concentrates on high-frequency regions where "
        "the perturbation was injected."
    )
