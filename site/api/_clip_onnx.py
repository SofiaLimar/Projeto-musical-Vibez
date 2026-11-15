"""CLIP ViT-B/32 (só o encoder de imagem) em ONNX, sem PyTorch.

O pré-processamento repete o CLIPProcessor do transformers:
redimensiona o lado menor para 224 (bicúbico), corta o centro 224×224,
divide por 255 e normaliza pela média/desvio do CLIP.
"""
from pathlib import Path

import numpy as np
from PIL import Image

TAMANHO = 224
MEDIA = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
DESVIO = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocessar(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    if w <= h:
        nw, nh = TAMANHO, int(TAMANHO * h / w)
    else:
        nw, nh = int(TAMANHO * w / h), TAMANHO
    img = img.resize((nw, nh), Image.BICUBIC)
    esq, topo = int((nw - TAMANHO) / 2), int((nh - TAMANHO) / 2)
    img = img.crop((esq, topo, esq + TAMANHO, topo + TAMANHO))
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEDIA) / DESVIO
    return x.transpose(2, 0, 1)  # (3, 224, 224)


class ClipImagem:
    def __init__(self, caminho: Path):
        import onnxruntime as ort

        opcoes = ort.SessionOptions()
        opcoes.intra_op_num_threads = 2
        self.sessao = ort.InferenceSession(str(caminho), opcoes, providers=["CPUExecutionProvider"])
        self.entrada = self.sessao.get_inputs()[0]
        self.tipo = np.float16 if "float16" in self.entrada.type else np.float32

    def embedding(self, img: Image.Image) -> np.ndarray:
        x = preprocessar(img)[None].astype(self.tipo)
        e = self.sessao.run(["image_embeds"], {self.entrada.name: x})[0][0].astype(np.float32)
        return e / np.linalg.norm(e)
