import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SITE_PACKAGES = ROOT / "venv" / "Lib" / "site-packages"
LOG_FILE = ROOT / "server.log"
ERROR_LOG = ROOT / "server.err"

if SITE_PACKAGES.exists():
    sys.path.insert(0, str(SITE_PACKAGES))

try:
    from app import app
except Exception:
    LOG_FILE.write_text(traceback.format_exc(), encoding="utf-8")
    raise


if __name__ == "__main__":
    try:
        with LOG_FILE.open("a", encoding="utf-8") as stdout, ERROR_LOG.open("a", encoding="utf-8") as stderr:
            sys.stdout = stdout
            sys.stderr = stderr
            print("Starting psychAI on http://127.0.0.1:5000", flush=True)
            app.run(host="127.0.0.1", port=5000)
            print("Server stopped without an exception.", flush=True)
    except Exception:
        LOG_FILE.write_text(traceback.format_exc(), encoding="utf-8")
        raise
