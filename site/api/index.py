"""
API do vibez na Vercel: foto -> playlist.

O algoritmo fica em _vibez/recomendar.py (única cópia, também usada pelo notebook).
Aqui ficam só o CLIP, a leitura da foto e o contrato da seção 4 do README.
O CLIP roda em ONNX (sem PyTorch) e reproduz o PyTorch exatamente
(validado em servidor/validar_onnx.py: r = 1,0000).

Dados em _dados/:
  modelo.json   Ridge como pesos W, b + media_prev/desvio_prev (Colab, seção 5b)
  musicas.json  as 2.812 músicas filtradas, na régua z (Colab, seção 5b)
  textos.npz    embeddings de texto do CLIP (servidor/preparar_textos.py)
O encoder de imagem (352 MB) não cabe no pacote: é baixado do Hugging Face
para /tmp quando a função acorda.
"""
import io
import json
import os
import sys
import threading
import urllib.request
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from _clip_onnx import ClipImagem  # noqa: E402
from _vibez.recomendar import GENEROS_SITE, VERSOES, Recomendador, SemMusicas  # noqa: E402

MAX_BYTES = 10 * 1024 * 1024

# encoder de imagem fixado numa versão exata do repositório público
URL_CLIP = ("https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
            "6b7c523703ae26e39be4b03774966f8571c7f523/onnx/vision_model.onnx")
TAMANHO_CLIP = 351_685_709
ARQ_CLIP = Path(os.environ.get("CLIP_ONNX", "/tmp/clip_visao.onnx"))

# ---------- dados (leves; carregam na importação) ----------
rec = Recomendador.da_pasta(AQUI / "_dados")

# ---------- CLIP (pesado; carrega na 1ª chamada) ----------
_clip = None
_trava = threading.Lock()


def clip():
    global _clip
    with _trava:
        if _clip is None:
            if not ARQ_CLIP.exists() or ARQ_CLIP.stat().st_size != TAMANHO_CLIP:
                tmp = ARQ_CLIP.with_suffix(".parcial")
                with urllib.request.urlopen(URL_CLIP, timeout=50) as r, open(tmp, "wb") as f:
                    while bloco := r.read(1 << 22):
                        f.write(bloco)
                if tmp.stat().st_size != TAMANHO_CLIP:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError("Download do CLIP incompleto.")
                tmp.rename(ARQ_CLIP)
            _clip = ClipImagem(ARQ_CLIP)
    return _clip


def playlist(img, generos, versao):
    try:
        return rec.playlist(clip().embedding(img), generos, versao)
    except SemMusicas as erro:
        raise HTTPException(422, str(erro))


# ---------- API (site e API no mesmo domínio: não precisa de CORS) ----------
app = FastAPI(title="vibez")


@app.get("/api/saude")
async def saude():
    # o site chama ao abrir: já baixa e carrega o CLIP antes da primeira foto
    await run_in_threadpool(clip)
    return {"ok": True, "musicas": len(rec.musicas)}


@app.post("/api/playlist")
async def criar_playlist(foto: UploadFile = File(...), generos: str = Form("[]"), versao: str = Form("atual")):
    # versao: "atual" (o site) ou "clip" (só para a avaliação do Passo 4, docs/avaliacao.md)
    if versao not in VERSOES:
        raise HTTPException(400, f"Versão desconhecida: {versao}")
    try:
        lista = json.loads(generos)
        assert isinstance(lista, list)
    except (ValueError, AssertionError):
        raise HTTPException(400, "Campo 'generos' inválido.")
    invalidos = [g for g in lista if g not in GENEROS_SITE]
    if invalidos:
        raise HTTPException(400, f"Gêneros desconhecidos: {', '.join(map(str, invalidos))}")

    conteudo = await foto.read(MAX_BYTES + 1)
    if len(conteudo) > MAX_BYTES:
        raise HTTPException(413, "A imagem passa de 10 MB.")
    try:
        img = Image.open(io.BytesIO(conteudo))
        img = ImageOps.exif_transpose(img).convert("RGB")   # fotos de celular vêm giradas no EXIF
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Não consegui abrir a imagem. Use JPG, PNG ou WEBP.")
    # a foto fica só na memória; nada é gravado
    return await run_in_threadpool(playlist, img, lista, versao)
