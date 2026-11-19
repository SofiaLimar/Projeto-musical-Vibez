"""
contar_rotulos.py — passo 1 do projeto playlist a partir de foto.

Baixa até N músicas por rótulo do MuSe (tag.getTopTracks do Last.fm), contando de verdade
quantas o Last.fm entrega. NÃO usa o atributo total/totalPages da API: no teste com
"energetic", a API informou total=37 e entregou 100 músicas.

Para de paginar quando: a página vem vazia, a página só traz músicas repetidas, ou atinge N.

Uso (na raiz do repositório):
  python pipeline/contar_rotulos.py --por-rotulo 1000
"""
import argparse, ast, json, os, sys, time
from pathlib import Path

import pandas as pd
import requests

API = "https://ws.audioscrobbler.com/2.0/"
CHAVE = os.environ.get("LASTFM_API_KEY")
PAUSA = 0.25                      # ~4 requisições/s
CACHE = Path("data/cache_lastfm")


def rotulos_muse(caminho):
    df = pd.read_csv(caminho, usecols=["seeds"])
    rotulos = set()
    for s in df["seeds"].dropna():
        for r in ast.literal_eval(s):
            rotulos.add(r.strip())
    return sorted(rotulos)


def pedir(rotulo, pagina, limite):
    params = dict(method="tag.gettoptracks", tag=rotulo, page=pagina,
                  limit=limite, api_key=CHAVE, format="json")
    for tentativa in range(5):
        try:
            r = requests.get(API, params=params, timeout=30)
            if r.status_code == 200:
                d = r.json()
                if "error" in d:
                    if d["error"] in (8, 11, 16, 29):   # erros temporários / limite de taxa
                        time.sleep(5 * (tentativa + 1))
                        continue
                    raise RuntimeError(f"{rotulo}: erro {d['error']} — {d.get('message')}")
                time.sleep(PAUSA)
                return d
        except requests.RequestException:
            pass
        time.sleep(5 * (tentativa + 1))
    raise RuntimeError(f"{rotulo}: falhou após 5 tentativas")


def baixar_rotulo(rotulo, por_rotulo, por_pagina):
    arq = CACHE / f"{rotulo.replace('/', '_')}_{por_rotulo}.json"
    if arq.exists():
        return json.loads(arq.read_text())

    faixas, vistas, pagina, motivo = [], set(), 1, "limite"
    while len(faixas) < por_rotulo:
        lote = pedir(rotulo, pagina, por_pagina)["tracks"].get("track", [])
        if isinstance(lote, dict):
            lote = [lote]
        if not lote:
            motivo = "pagina_vazia"
            break
        novas = 0
        for f in lote:
            k = (f["artist"]["name"].casefold().strip(), f["name"].casefold().strip())
            if k in vistas:
                continue
            vistas.add(k)
            novas += 1
            faixas.append(dict(artista=f["artist"]["name"], titulo=f["name"],
                               mbid=f.get("mbid", ""),
                               posicao=int(f.get("@attr", {}).get("rank", len(faixas) + 1))))
        if novas == 0:
            motivo = "so_repetidas"
            break
        pagina += 1

    res = dict(rotulo=rotulo, paginas_lidas=pagina, motivo_parada=motivo,
               faixas=faixas[:por_rotulo])
    arq.write_text(json.dumps(res, ensure_ascii=False))
    return res


def main(rotulos, por_rotulo, por_pagina):
    CACHE.mkdir(parents=True, exist_ok=True)
    saida = f"data/lista_rotulos_{por_rotulo}.csv"
    linhas, conferencia = [], []
    for i, r in enumerate(rotulos, 1):
        res = baixar_rotulo(r, por_rotulo, por_pagina)
        conferencia.append(dict(rotulo=r, entregues=len(res["faixas"]),
                                paginas_lidas=res["paginas_lidas"],
                                motivo_parada=res["motivo_parada"]))
        for f in res["faixas"]:
            linhas.append(dict(rotulo=r, **f))
        print(f"[{i}/{len(rotulos)}] {r}: {len(res['faixas'])} ({res['motivo_parada']})")

    df = pd.DataFrame(linhas)
    df.to_csv(saida, index=False)
    conf = pd.DataFrame(conferencia).sort_values("entregues")
    conf.to_csv(saida.replace(".csv", "_conferencia.csv"), index=False)

    e = conf["entregues"]
    chave = df["artista"].str.casefold().str.strip() + " — " + df["titulo"].str.casefold().str.strip()
    print("\n=== Resumo ===")
    print(f"rótulos: {len(conf)}")
    print(f"com 0 músicas: {(e == 0).sum()} | < 100: {(e < 100).sum()} | "
          f"< 500: {(e < 500).sum()} | atingiram o limite ({por_rotulo}): {(e >= por_rotulo).sum()}")
    print(f"mediana por rótulo: {e.median():.0f}")
    print(f"linhas (rótulo × música): {len(df)}")
    print(f"músicas únicas (artista+título, sem diferenciar maiúsculas): {chave.nunique()}")
    print("motivo de parada:", conf["motivo_parada"].value_counts().to_dict())
    print(f"salvo em {saida}")


if __name__ == "__main__":
    if not CHAVE:
        sys.exit("Defina LASTFM_API_KEY no terminal antes de rodar.")
    p = argparse.ArgumentParser()
    p.add_argument("--muse", default="data/muse_v3.csv")
    p.add_argument("--por-rotulo", type=int, default=1000)
    p.add_argument("--por-pagina", type=int, default=200)
    a = p.parse_args()

    rotulos = rotulos_muse(a.muse)
    print(f"{len(rotulos)} rótulos lidos de {a.muse}")
    main(rotulos, a.por_rotulo, a.por_pagina)