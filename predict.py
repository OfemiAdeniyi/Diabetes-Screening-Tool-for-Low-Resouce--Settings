import os
import requests
import joblib

# -----------------------------
# Model Metadata
# -----------------------------
MODEL_VERSION = "1.0.0"

# -----------------------------
# Google Drive Direct Download URLs
# -----------------------------
REDUCED_MODEL_URL = (
    "https://drive.google.com/uc?id=1o0dPggLeTrdVy7dESmsvP6IDk_Q5SeGF&export=download"
)

THRESHOLD_URL = (
    "https://drive.google.com/uc?id=1poS0OD5Z6lhl1pfgg1AyG_GmHGEweYu8&export=download"
)

# -----------------------------
# Local Artifact Paths
# -----------------------------
ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "reduced_rf_model.pkl")
THRESHOLD_PATH = os.path.join(ARTIFACT_DIR, "screening_threshold.pkl")


# -----------------------------
# Utilities
# -----------------------------
def download_file(url: str, path: str):
    """
    Download a file from Google Drive if it does not exist locally.
    """
    if os.path.exists(path):
        return

    print(f"⬇️ Downloading artifact → {path}")

    response = requests.get(url)
    response.raise_for_status()

    with open(path, "wb") as f:
        f.write(response.content)

    print(f"✅ Download completed → {path}")


# -----------------------------
# Load Artifacts (Executed Once)
# -----------------------------
os.makedirs(ARTIFACT_DIR, exist_ok=True)

download_file(REDUCED_MODEL_URL, MODEL_PATH)
download_file(THRESHOLD_URL, THRESHOLD_PATH)

try:
    reduced_rf_model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

try:
    screening_threshold = joblib.load(THRESHOLD_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load threshold: {e}")
