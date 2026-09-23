"""
Central configuration. Change values here rather than editing scripts.
"""
import os
import torch

# ---------------- Paths ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
ADV_DIR = os.path.join(ROOT, "adv_data")
RESULTS_DIR = os.path.join(ROOT, "results")

for _d in (DATA_DIR, CKPT_DIR, ADV_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

CLASSIFIER_CKPT = os.path.join(CKPT_DIR, "classifier_gtsrb.pt")
DETECTOR_CKPT = os.path.join(CKPT_DIR, "detector_gtsrb.pt")

# ---------------- Device ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Dataset ----------------
DATASET = "gtsrb"          # "gtsrb" or "cifar10"
IMG_SIZE = 32
NUM_CLASSES = {"gtsrb": 43, "cifar10": 10}

# Normalisation happens INSIDE the model (see models.NormalizedModel), so all
# DataLoaders yield tensors in [0, 1]. This is required for torchattacks to
# operate in valid pixel space.
MEAN = (0.3403, 0.3121, 0.3214)   # GTSRB channel means
STD = (0.2724, 0.2608, 0.2669)

# ---------------- Training ----------------
BATCH_SIZE = 128
EPOCHS_CLASSIFIER = 15
EPOCHS_DETECTOR = 12
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

# ---------------- Attacks ----------------
# eps values are in [0,1] pixel scale. 8/255 is the standard Linf budget.
EPS = 8 / 255
PGD_ALPHA = 2 / 255
PGD_STEPS = 10
CW_STEPS = 50
CW_C = 1.0
SQUARE_QUERIES = 2000

ATTACKS = ["fgsm", "pgd", "cw", "square"]

# How many test images to attack per attack type (C&W and Square are slow —
# lower this if you are on CPU).
N_ATTACK_SAMPLES = 2000

# ---------------- Adaptive attack ----------------
ADAPTIVE_LAMBDA = 1.0      # weight on the detector-evasion term
ADAPTIVE_STEPS = 20

# ---------------- Feature Squeezing baseline ----------------
SQUEEZE_BIT_DEPTH = 4
SQUEEZE_MEDIAN_KERNEL = 2
TARGET_FPR = 0.05          # pick threshold at 5% false-positive rate
