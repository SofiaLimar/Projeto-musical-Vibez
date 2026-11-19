"""Pré-calcula os embeddings de texto do CLIP (frases de cena + rótulos MuSe).
Assim a Vercel só precisa do encoder de imagem.
Uso (na raiz do repositório): python ferramentas/preparar_textos.py   (gera site/api/_dados/textos.npz)
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "site" / "api" / "_dados"
CENAS_FESTA = ["a photo of a party", "a photo of people dancing at a nightclub",
               "a photo of a celebration with friends", "a photo of a concert crowd"]
CENAS_OUTRAS = ["a photo of a landscape", "a photo of a beach", "a portrait of a person",
                "a photo of a city street", "a photo of nature", "a photo of food",
                "a photo of a room", "a selfie", "a photo of animals"]

modelo = json.loads((DADOS / "modelo.json").read_text())
musicas = json.loads((DADOS / "musicas.json").read_text())
rotulos = sorted({r for m in musicas for r in m["rotulos"]})

clip = CLIPModel.from_pretrained(modelo["clip"]).eval()
proc = CLIPProcessor.from_pretrained(modelo["clip"])


@torch.no_grad()
def emb_texto(frases):
    x = proc(text=frases, return_tensors="pt", padding=True)
    t = clip.text_projection(clip.text_model(input_ids=x["input_ids"],
                                             attention_mask=x["attention_mask"]).pooler_output)
    return torch.nn.functional.normalize(t, dim=-1).numpy().astype(np.float32)


np.savez_compressed(DADOS / "textos.npz",
                    cenas=emb_texto(CENAS_FESTA + CENAS_OUTRAS), n_festa=len(CENAS_FESTA),
                    rotulos=emb_texto([f"a photo with a {r} mood" for r in rotulos]),
                    nomes_rotulos=np.array(rotulos))
print(len(rotulos), "rótulos;", DADOS / "textos.npz")
