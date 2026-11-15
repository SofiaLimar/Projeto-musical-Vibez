"""
API do vibez na Vercel: foto -> playlist.

Segue o algoritmo do README (seção 1) e o contrato da seção 4.
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
from collections import Counter
import json
import re
import math
import os
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from _clip_onnx import ClipImagem  # noqa: E402

# ---------- parâmetros (README, seção 1) ----------
TAMANHO_PLAYLIST = 15
CANDIDATOS_POR_VAGA = 6
ALFA = 2          # peso da popularidade (varredura OASIS: 73% das músicas no top 10% de pop, dist p90 0,47)
MAX_POR_ARTISTA = 1
CANDIDATOS_GENERO = 30   # modo normal sem gênero escolhido: playlist no gênero dominante + 1 vizinho (15/11)

GEN_FESTA = {"Pop", "Dance", "Electro", "Rap/Hip Hop", "R&B"}   # gêneros do Deezer
LIMIAR_FESTA = 0.5
ALVO_FESTA = (0.5, 1.0)
BETA_FESTA = 0.5
TOP_ROTULOS = 15

# ---------- gênero (README, seção 2) ----------
GENEROS_SITE = ["Pop", "Rock", "Alternativo", "R&B", "Rap/Hip Hop", "Eletrônica", "Trilhas sonoras"]
VIZINHOS = {
    "Pop": ["R&B", "Alternativo"],
    "Rock": ["Alternativo", "Pop"],
    "Alternativo": ["Rock", "Pop"],
    "R&B": ["Pop", "Rap/Hip Hop"],
    "Rap/Hip Hop": ["R&B", "Pop"],
    "Eletrônica": ["Pop", "Alternativo"],
    "Trilhas sonoras": ["Alternativo", "Pop"],
}

MAX_BYTES = 10 * 1024 * 1024

# encoder de imagem fixado numa versão exata do repositório público
URL_CLIP = ("https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
            "6b7c523703ae26e39be4b03774966f8571c7f523/onnx/vision_model.onnx")
TAMANHO_CLIP = 351_685_709
ARQ_CLIP = Path(os.environ.get("CLIP_ONNX", "/tmp/clip_visao.onnx"))

# ---------- dados (leves; carregam na importação) ----------
DADOS = AQUI / "_dados"
modelo = json.loads((DADOS / "modelo.json").read_text())
W = np.array(modelo["W"], dtype=np.float32)            # (2, 512)
B = np.array(modelo["b"], dtype=np.float32)
MEDIA_PREV = np.array(modelo["media_prev"], dtype=np.float32)
DESVIO_PREV = np.array(modelo["desvio_prev"], dtype=np.float32)

musicas = json.loads((DADOS / "musicas.json").read_text())
Zm = np.array([[m["zv"], m["ze"]] for m in musicas], dtype=np.float32)
POP = np.array([m["pop"] for m in musicas], dtype=np.float32)
GEN_SITE = np.array([m["genero"] for m in musicas])
GEN_DEEZER = np.array([m["genero_deezer"] for m in musicas])
# artista-base: "Bob Marley & The Wailers" e "Bob Marley" contam como o mesmo artista
ARTISTA = [re.split(r" & | feat\.? | ft\.? |, | and ", m["artista"].lower())[0].strip() for m in musicas]
DEEZER_ID = [m["deezer_id"] for m in musicas]

textos = np.load(DADOS / "textos.npz")
T_CENAS = textos["cenas"]
N_FESTA = int(textos["n_festa"])
T_ROT = textos["rotulos"]
ROTULOS = [str(r) for r in textos["nomes_rotulos"]]
IDX_ROT = {r: i for i, r in enumerate(ROTULOS)}
LISTAS_ROT = [[IDX_ROT[r] for r in m["rotulos"] if r in IDX_ROT] for m in musicas]

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


# ---------- algoritmo ----------
def prob_festa(e):
    s = 100 * (T_CENAS @ e)
    p = np.exp(s - s.max())
    p /= p.sum()
    return float(p[:N_FESTA].sum())


def nota_rotulos(e):
    s = T_ROT @ e
    top = np.argsort(-s)[:TOP_ROTULOS]
    w = np.zeros(len(ROTULOS))
    ex = np.exp(100 * (s[top] - s[top].max()))
    w[top] = ex / ex.max()
    tag = np.array([max((w[i] for i in ls), default=0.0) for ls in LISTAS_ROT])
    return tag, [ROTULOS[i] for i in top[:6]]


def escolher(nota, n=TAMANHO_PLAYLIST):
    escolhidas, por_artista, ids = [], {}, set()
    for i in np.argsort(nota):
        if not np.isfinite(nota[i]):
            break
        a = ARTISTA[i]
        if por_artista.get(a, 0) >= MAX_POR_ARTISTA or DEEZER_ID[i] in ids:
            continue
        ids.add(DEEZER_ID[i])
        por_artista[a] = por_artista.get(a, 0) + 1
        escolhidas.append(int(i))
        if len(escolhidas) == n:
            break
    return escolhidas


def mesmo_quadrante(z):
    return (np.sign(Zm[:, 0]) == np.sign(z[0])) & (np.sign(Zm[:, 1]) == np.sign(z[1]))


def generos_com_fallback(generos, z, n=TAMANHO_PLAYLIST):
    """Se os gêneros escolhidos têm < n músicas no quadrante da foto, completa com vizinhos."""
    usados = list(generos)
    quad = mesmo_quadrante(z)
    acrescentados = []
    for viz in [v for g in generos for v in VIZINHOS[g]]:
        if (np.isin(GEN_SITE, usados) & quad).sum() >= n:
            break
        if viz not in usados:
            usados.append(viz)
            acrescentados.append(viz)
    aviso = None
    if acrescentados:
        aviso = (f"Poucas músicas de {' e '.join(generos)} combinam com o clima desta foto; "
                 f"completamos com {' e '.join(acrescentados)}.")
    return usados, aviso


def nome_playlist(z, festa):
    if festa:
        return "Clima de festa"
    alegre, agitada = z[0] >= 0, z[1] >= 0
    return {(True, True): "Alto astral", (True, False): "Leve e ensolarada",
            (False, True): "Intensa e inquieta", (False, False): "Calma e melancólica"}[(alegre, agitada)]


def percentil(zv):
    return round(50 * (1 + math.erf(float(zv) / math.sqrt(2))), 1)


def playlist(img, generos):
    e = clip().embedding(img)
    p = W @ e + B
    z = (p - MEDIA_PREV) / DESVIO_PREV
    pf = prob_festa(e)
    festa = pf >= LIMIAR_FESTA
    tag, rotulos = nota_rotulos(e)
    aviso = None
    n = TAMANHO_PLAYLIST

    if festa:
        alvo = np.array([max(z[0], ALVO_FESTA[0]), max(z[1], ALVO_FESTA[1])])
        gen_ok = np.isin(GEN_SITE, generos) if generos else np.isin(GEN_DEEZER, list(GEN_FESTA))
        ok = gen_ok & (Zm[:, 0] > 0) & (Zm[:, 1] > 0.3)
        if ok.sum() < n * 2:
            ok = gen_ok
        dist = np.linalg.norm(Zm - alvo, axis=1)
        nota = np.where(ok, dist - ALFA * POP - BETA_FESTA * tag, np.inf)
    else:
        if generos:
            usados, aviso = generos_com_fallback(generos, z)
            ok = np.isin(GEN_SITE, usados)
        else:
            ok = np.ones(len(musicas), bool)
        dist = np.linalg.norm(Zm - z, axis=1)

        def nota_perto(ok, k):
            k = min(int(ok.sum()), k)
            perto = np.argsort(np.where(ok, dist, np.inf))[:k]
            nota = np.full(len(musicas), np.inf)
            nota[perto] = dist[perto] - ALFA * POP[perto]
            return nota

        nota = nota_perto(ok, n * CANDIDATOS_POR_VAGA)   # as 90 mais próximas
        if not generos:
            # sem gênero escolhido: usa o gênero mais comum da playlist provisória + 1 vizinho,
            # para não misturar reggae, k-pop e emo quando a foto cai no centro do mapa
            contagem = Counter(g for g in GEN_SITE[escolher(nota)] if g in VIZINHOS)
            if contagem:
                dom = contagem.most_common(1)[0][0]
                nota = nota_perto(np.isin(GEN_SITE, [dom, VIZINHOS[dom][0]]), CANDIDATOS_GENERO)

    escolhidas = escolher(nota)
    if not escolhidas:
        raise HTTPException(422, "Não há músicas suficientes nesses gêneros.")

    return {
        "nome": nome_playlist(z, festa),
        "valencia_pct": percentil(z[0]),
        "energia_pct": percentil(z[1]),
        "z": [round(float(z[0]), 3), round(float(z[1]), 3)],
        "modo_festa": bool(festa),
        "prob_festa": round(pf, 3),
        "rotulos": rotulos,
        "aviso": aviso,
        "link_playlist": None,
        "faixas": [{
            "titulo": musicas[i]["titulo"],
            "artista": musicas[i]["artista"],
            "genero": musicas[i]["genero"] or musicas[i]["genero_deezer"],
            "deezer_id": musicas[i]["deezer_id"],
            "previa_url": None,   # o site busca a prévia no Deezer na hora
        } for i in escolhidas],
    }


# ---------- API (site e API no mesmo domínio: não precisa de CORS) ----------
app = FastAPI(title="vibez")


@app.get("/api/saude")
async def saude():
    # o site chama ao abrir: já baixa e carrega o CLIP antes da primeira foto
    await run_in_threadpool(clip)
    return {"ok": True, "musicas": len(musicas)}


@app.post("/api/playlist")
async def criar_playlist(foto: UploadFile = File(...), generos: str = Form("[]")):
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
    return await run_in_threadpool(playlist, img, lista)
