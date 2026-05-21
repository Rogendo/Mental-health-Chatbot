import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SITE_PACKAGES = ROOT / "venv" / "Lib" / "site-packages"
LOG_FILE = ROOT / "server.log"

if SITE_PACKAGES.exists():
    sys.path.insert(0, str(SITE_PACKAGES))

try:
    from app import app
except Exception:
    LOG_FILE.write_text(traceback.format_exc(), encoding="utf-8")
    raise


if __name__ == "__main__":
    try:
        LOG_FILE.write_text("Starting psychAI on http://127.0.0.1:5000\n", encoding="utf-8")
        app.run(host="127.0.0.1", port=5000)
        LOG_FILE.write_text("Server stopped without an exception.\n", encoding="utf-8")
    except Exception:
        LOG_FILE.write_text(traceback.format_exc(), encoding="utf-8")
        raise
