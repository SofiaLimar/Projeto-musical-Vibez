"""
API do vibez na Vercel: foto -> playlist.

O algoritmo fica em _vibez/recomendar.py (única cópia, também usada pelo notebook).
Aqui ficam só o CLIP, a leitura da foto e o contrato da seção 4 de docs/decisoes.md.
O CLIP roda em ONNX (sem PyTorch) e reproduz o PyTorch exatamente
(validado em ferramentas/validar_onnx.py: r = 1,0000).

Dados em _dados/:
  modelo.json   Ridge como pesos W, b + media_prev/desvio_prev (Colab, seção 5b)
  musicas.json  as 2.812 músicas filtradas, na régua z (Colab, seção 5b)
  textos.npz    embeddings de texto do CLIP (ferramentas/preparar_textos.py)
O encoder de imagem (352 MB) não cabe no pacote: é baixado do Hugging Face
para /tmp quando a função acorda.
"""
import io
import json
import os
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from _clip_onnx import ClipImagem, baixar  # noqa: E402
from _vibez.recomendar import GENEROS_SITE, VERSOES, Recomendador, SemMusicas  # noqa: E402

MAX_BYTES = 10 * 1024 * 1024

# encoder de imagem fixado numa versão exata do repositório público.
# Tamanho e sha256 conferidos na API da Hugging Face (lfs.size / lfs.oid) em 18/11/2026.
BASE_CLIP = ("https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
             "6b7c523703ae26e39be4b03774966f8571c7f523/onnx/")
MODELOS_CLIP = {
    "fp32": ("vision_model.onnx", 351_685_709,
             "fd6e1402a588279d1723c7534d4bcba5bc0b14b47dfab0e46f8c47b8270d7d40"),
    # 4× menor, mas REPROVADO em 18/11/2026 no ferramentas/validar_onnx.py (200 fotos do OASIS):
    # r humor 0,974 (meta > 0,98), r energia 0,983, cosseno mín. 0,772, dif. máx. de previsão 1,18. Não usar.
    "quantizado": ("vision_model_quantized.onnx", 89_117_001,
                   "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299"),
}
MODELO_CLIP = "fp32"
NOME_CLIP, TAMANHO_CLIP, SHA256_CLIP = MODELOS_CLIP[MODELO_CLIP]
URL_CLIP = BASE_CLIP + NOME_CLIP
ARQ_CLIP = Path(os.environ.get("CLIP_ONNX", f"/tmp/clip_{MODELO_CLIP}.onnx"))
PRAZO_DOWNLOAD = 45   # segundos para o download inteiro (a função tem maxDuration de 60 s)

# ---------- dados (leves; carregam na importação) ----------
rec = Recomendador.da_pasta(AQUI / "_dados")

# ---------- CLIP (pesado; carrega na 1ª chamada) ----------
_clip = None
_trava = threading.Lock()


def clip():
    global _clip
    with _trava:
        if _clip is None:
            # o arquivo em /tmp só existe se já passou pela conferência de sha256 em baixar()
            if not ARQ_CLIP.exists() or ARQ_CLIP.stat().st_size != TAMANHO_CLIP:
                baixar(URL_CLIP, ARQ_CLIP, TAMANHO_CLIP, SHA256_CLIP, PRAZO_DOWNLOAD)
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


@app.get("/api/generos")
async def generos_do_site():
    # a lista de gêneros fica só em _vibez/recomendar.py; o site pede aqui ao abrir
    return {"generos": GENEROS_SITE}


@app.post("/api/playlist")
async def criar_playlist(foto: UploadFile = File(...), generos: str = Form("[]"), versao: str = Form("atual")):
    # versao: "atual" (o site) ou "clip" (só para a avaliação do Passo 4, docs/avaliacao.md)
    if versao not in VERSOES:
        raise HTTPException(400, f"Versão desconhecida: {versao}")
    # validação explícita (um assert some quando o Python roda com -O)
    try:
        lista = json.loads(generos)
    except ValueError:
        lista = None
    if not isinstance(lista, list):
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
