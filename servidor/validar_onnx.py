"""
Confere se o CLIP em ONNX (sem PyTorch) dá as mesmas previsões do CLIP em PyTorch.
Meta do README: r > 0,98 nas previsões de humor/energia.

Uso: .venv/bin/python validar_onnx.py <pasta_onnx> <pasta_fotos>
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "site" / "api"))
from _clip_onnx import preprocessar  # mesmo pré-processamento que vai para a Vercel

pasta_onnx, pasta_fotos = Path(sys.argv[1]), Path(sys.argv[2])
modelo = json.loads((Path(__file__).resolve().parent.parent / "site" / "api" / "_dados" / "modelo.json").read_text())
W, B = np.array(modelo["W"]), np.array(modelo["b"])

fotos = sorted(p for p in pasta_fotos.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
imgs = [Image.open(p).convert("RGB") for p in fotos]
print(len(imgs), "fotos")

clip = CLIPModel.from_pretrained(modelo["clip"]).eval()
proc = CLIPProcessor.from_pretrained(modelo["clip"])
with torch.no_grad():
    x = proc(images=imgs, return_tensors="pt")
    e = clip.visual_projection(clip.vision_model(pixel_values=x["pixel_values"]).pooler_output)
    ref = torch.nn.functional.normalize(e, dim=-1).numpy()
    px_ref = x["pixel_values"].numpy()

px = np.stack([preprocessar(im) for im in imgs])
print(f"pré-processamento: dif. máx. de pixel {np.abs(px - px_ref).max():.4f}")

p_ref = ref @ W.T + B
for arq in sorted(pasta_onnx.rglob("vision_model*.onnx")):
    s = ort.InferenceSession(str(arq), providers=["CPUExecutionProvider"])
    entrada = s.get_inputs()[0]
    dados = px.astype(np.float16 if "float16" in entrada.type else np.float32)
    out = s.run(["image_embeds"], {entrada.name: dados})[0].astype(np.float64)
    emb = out / np.linalg.norm(out, axis=1, keepdims=True)
    p = emb @ W.T + B
    cos = (emb * ref).sum(1)
    r = [np.corrcoef(p[:, k], p_ref[:, k])[0, 1] for k in range(2)]
    print(f"{arq.name:32s} {arq.stat().st_size/1e6:6.1f} MB | cosseno mín {cos.min():.4f} | "
          f"r humor {r[0]:.4f} energia {r[1]:.4f} | dif. máx. previsão {np.abs(p - p_ref).max():.3f}")
