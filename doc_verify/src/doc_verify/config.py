import os
from dotenv import load_dotenv

from ml_utils.paths import find_workspace_root

def ensure_directories():
    for dir_path in [
        DATA_DIR,
        TEMP_DIR,
        MODELS_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

load_dotenv()

WORKSPACE_ROOT = find_workspace_root(__file__)


DATA_DIR   = WORKSPACE_ROOT / "data"
TEMP_DIR   = DATA_DIR / "temp"
MODELS_DIR = WORKSPACE_ROOT / "models"


ACCOUNT        = os.getenv("AZURE_STORAGE_ACCOUNT")
ACCOUNT_KEY    = os.getenv("AZURE_STORAGE_KEY")
CONTAINER_NAME = os.getenv("DOC_VER_CONTAINER_NAME")

ensure_directories()