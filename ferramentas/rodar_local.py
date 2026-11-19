"""Roda site + API juntos no Mac, como na Vercel.
Uso: CLIP_ONNX=<caminho do vision_model.onnx> .venv/bin/uvicorn rodar_local:app --port 8010
(sem CLIP_ONNX, o modelo é baixado para /tmp na primeira chamada)
"""
import sys
from pathlib import Path

from fastapi.staticfiles import StaticFiles

SITE = Path(__file__).resolve().parent.parent / "site" / "public"
sys.path.insert(0, str(SITE.parent / "api"))
from index import app  # noqa: E402

app.mount("/", StaticFiles(directory=SITE, html=True), name="site")
