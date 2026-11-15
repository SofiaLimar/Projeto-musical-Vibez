# ============================================================================
# CÉLULA PARA COLAR NO FIM DO foto_parte2_playlist.ipynb (no Colab)
#
# Rode o notebook até a seção 5 (régua z-score), para existirem `musicas`
# (já limpa e filtrada por POP_MIN), `Zm` e `pac`. Depois rode esta célula.
#
# Ela grava em  MyDrive/mapa-sonoro/servidor/dados/ :
#   - musicas.json : as ~2.812 músicas com zv, ze, pop, gênero e rótulos
#   - modelo.json  : o Ridge como pesos (W, b) + media_prev/desvio_prev
# Baixe os dois para  mapa-sonoro/servidor/dados/  no Mac.
#
# O Ridge vira pesos para o servidor não depender da versão do scikit-learn
# (um .joblib só abre com a mesma versão que o gravou).
# ============================================================================
import json
import numpy as np

SAIDA = PASTA / "servidor" / "dados"
SAIDA.mkdir(parents=True, exist_ok=True)

# ---------- 1. modelo: StandardScaler + Ridge  ->  p = W @ e + b ----------
modelo = pac["modelo"]
etapas = list(modelo.named_steps.values()) if hasattr(modelo, "named_steps") else [modelo]
escala, ridge = None, etapas[-1]
for et in etapas[:-1]:
    if hasattr(et, "scale_"):
        escala = et
    else:
        raise ValueError(f"Etapa inesperada no modelo: {et!r}")

coef = np.atleast_2d(ridge.coef_)          # (2, 512)
inter = np.atleast_1d(ridge.intercept_)    # (2,)
if escala is not None:
    media = escala.mean_ if escala.with_mean else 0.0
    W = coef / escala.scale_
    b = inter - (coef * media / escala.scale_).sum(axis=1)
else:
    W, b = coef, inter

# confere: os pesos têm que dar exatamente a mesma previsão do modelo
X_teste = np.load(PASTA / "modelos" / "clip_oasis_embeddings.npy")
dif = np.abs(X_teste @ W.T + b - modelo.predict(X_teste)).max()
print(f"diferença máxima entre pesos e modelo: {dif:.2e}")
assert dif < 1e-6, "Os pesos não reproduzem o modelo; não use este export."

json.dump({
    "clip": pac["clip"],
    "W": W.tolist(),
    "b": b.tolist(),
    "media_prev": np.asarray(pac["media_prev"]).tolist(),
    "desvio_prev": np.asarray(pac["desvio_prev"]).tolist(),
    "validacao": str(pac.get("validacao", "")),
}, open(SAIDA / "modelo.json", "w"))

# ---------- 2. músicas já filtradas, na régua z ----------
PARA_SITE = {"Pop": "Pop", "Rock": "Rock", "Alternativo": "Alternativo", "R&B": "R&B",
             "Rap/Hip Hop": "Rap/Hip Hop", "Electro": "Eletrônica", "Dance": "Eletrônica",
             "Filmes/Games": "Trilhas sonoras"}
assert len(musicas) == len(Zm), "Rode de novo a seção 5 antes desta célula."
itens = []
for (_, m), (zv, ze) in zip(musicas.iterrows(), Zm):
    itens.append({
        "deezer_id": int(m.deezer_id),
        "artista": str(m.artista),
        "titulo": str(m.titulo),
        "genero_deezer": str(m.genero or ""),
        "genero": PARA_SITE.get(str(m.genero or ""), ""),
        "rotulos": [r for r in str(m.rotulos or "").split("|") if r and r != "nan"],
        "zv": round(float(zv), 4),
        "ze": round(float(ze), 4),
        "pop": round(float(m["pop"]), 4),
    })
json.dump(itens, open(SAIDA / "musicas.json", "w"), ensure_ascii=False)

import sklearn, transformers, torch
print(f"{len(itens)} músicas exportadas (esperado: ~2.812)")
print("versões no Colab: sklearn", sklearn.__version__, "| transformers", transformers.__version__, "| torch", torch.__version__)
print("arquivos em", SAIDA)
