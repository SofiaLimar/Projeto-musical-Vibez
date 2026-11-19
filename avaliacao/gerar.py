"""
Gera a página de avaliação humana (docs/avaliacao.md).

Para cada foto de --fotos, pede a playlist a cada versão da API (--versao nome=URL)
e grava, em --saida:
  avaliar.html   a página para as avaliadoras (uma só, com fotos e playlists embutidas;
                 cega: só códigos sorteados, sem nome de versão)
  chave.json     código -> foto e versão, e a resposta completa da API. NÃO enviar às avaliadoras.

Uso (na raiz do repositório):
  python avaliacao/gerar.py --fotos avaliacao/fotos \
      --versao atual=https://vibez-azure.vercel.app
  # Passo 4: as duas versões vêm da mesma API; o nome (atual/clip) vai no campo "versao"
  python avaliacao/gerar.py --fotos avaliacao/fotos \
      --versao atual=https://vibez-azure.vercel.app --versao clip=https://vibez-azure.vercel.app
  # 2ª rodada (docs/avaliacao.md, seção 10): atual x mahalanobis, numa pasta de saída separada
  python avaliacao/gerar.py --fotos avaliacao/fotos --saida avaliacao/saida2 \
      --versao atual=https://vibez-azure.vercel.app --versao mahalanobis=https://vibez-azure.vercel.app
"""
import argparse
import base64
import io
import json
import secrets
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

AQUI = Path(__file__).resolve().parent
EXTENSOES = {".jpg", ".jpeg", ".png", ".webp"}


def ler_foto(caminho):
    img = Image.open(caminho)
    return ImageOps.exif_transpose(img).convert("RGB")   # fotos de celular vêm giradas no EXIF


def jpeg(img, lado_max, qualidade):
    img = img.copy()
    img.thumbnail((lado_max, lado_max))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=qualidade)
    return buf.getvalue()


def pedir_playlist(url, versao, nome, conteudo, tentativas=3):
    for t in range(tentativas):
        try:
            r = requests.post(f"{url}/api/playlist", files={"foto": (nome, conteudo, "image/jpeg")},
                              data={"generos": "[]", "versao": versao}, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code < 500:
                sys.exit(f"{url} respondeu {r.status_code} para {nome}: {r.text[:200]}")
        except requests.RequestException as erro:
            print(f"  {nome}: {erro}")
        time.sleep(5 * (t + 1))
    sys.exit(f"{url} não respondeu para {nome} depois de {tentativas} tentativas.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fotos", required=True, type=Path)
    ap.add_argument("--versao", required=True, action="append", metavar="NOME=URL")
    ap.add_argument("--saida", type=Path, default=AQUI / "saida")
    args = ap.parse_args()

    versoes = dict(v.split("=", 1) for v in args.versao)
    versoes = {nome: url.rstrip("/") for nome, url in versoes.items()}
    fotos = sorted(p for p in args.fotos.iterdir() if p.suffix.lower() in EXTENSOES)
    if not fotos:
        sys.exit(f"Nenhuma foto em {args.fotos}")
    print(f"{len(fotos)} fotos × {len(versoes)} versão(ões): {', '.join(versoes)}")

    for nome, url in versoes.items():   # acorda as funções (o CLIP é baixado na 1ª chamada)
        print(f"acordando {nome} ({url})...")
        requests.get(f"{url}/api/saude", timeout=120).raise_for_status()

    itens, chave = [], {"versoes": versoes, "codigos": {}}
    for k, caminho in enumerate(fotos, 1):
        img = ler_foto(caminho)
        envio = jpeg(img, 1600, 90)
        foto_id = f"f{k:02d}"
        playlists = []
        for nome, url in versoes.items():
            resp = pedir_playlist(url, nome, caminho.name, envio)
            codigo = secrets.token_hex(3)
            chave["codigos"][codigo] = {"foto": foto_id, "arquivo": caminho.name, "versao": nome, "resposta": resp}
            playlists.append({"codigo": codigo,
                              "faixas": [{"titulo": f["titulo"], "artista": f["artista"], "deezer_id": f["deezer_id"]}
                                         for f in resp["faixas"]]})
        itens.append({"foto": foto_id,
                      "imagem": "data:image/jpeg;base64," + base64.b64encode(jpeg(img, 640, 80)).decode(),
                      "playlists": playlists})
        print(f"{k}/{len(fotos)} {caminho.name}")

    args.saida.mkdir(parents=True, exist_ok=True)
    estudo = secrets.token_hex(4)
    dados = json.dumps({"estudo": estudo, "itens": itens}, ensure_ascii=False).replace("</", "<\\/")   # seguro dentro de <script>
    pagina = (AQUI / "modelo_pagina.html").read_text(encoding="utf-8").replace("__DADOS__", dados)
    (args.saida / "avaliar.html").write_text(pagina, encoding="utf-8")
    chave["estudo"] = estudo
    (args.saida / "chave.json").write_text(json.dumps(chave, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\npronto: {args.saida / 'avaliar.html'} (enviar) e {args.saida / 'chave.json'} (guardar, não enviar)")


if __name__ == "__main__":
    main()
