from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT / "models"      # the trained classifier, kept in git
VISION_DIR = MODELS_DIR / "vision"  # downloaded face and eye models
DATA_DIR = ROOT / "data"          # downloaded training data
INSTANCE_DIR = ROOT / "instance"  # the app's database and secret key

DATABASE = INSTANCE_DIR / "cyberbullying.db"
