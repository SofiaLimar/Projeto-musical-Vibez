"""
buscar_generos_albuns.py — gênero de cada álbum no Deezer.

Lê os album_deezer_id de data/popularidade.csv e grava data/generos_albuns.csv
(album_deezer_id, generos, status). "generos" vem separado por "|" (ex.: "Pop|Dance").
Retoma de onde parou.
"""
import csv, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from buscar_popularidade import esperar_vez, sessao, DEEZER

SAIDA = Path("data/generos_albuns.csv")


def consultar(alb_id):
    for tentativa in range(6):
        esperar_vez()
        try:
            dados = sessao.get(f"{DEEZER}/album/{alb_id}", timeout=15).json()
        except (requests.RequestException, ValueError):
            time.sleep(2 ** tentativa)
            continue
        erro = dados.get("error")
        if erro:
            if erro.get("code") == 4:
                time.sleep(5)
                continue
            return {"album_deezer_id": alb_id, "generos": "", "status": f"erro_{erro.get('code')}"}
        generos = [g.get("name", "") for g in (dados.get("genres") or {}).get("data", [])]
        return {"album_deezer_id": alb_id, "generos": "|".join(generos), "status": "ok"}
    return None


def main():
    ids = (pd.read_csv("data/popularidade.csv").album_deezer_id.dropna()
           .astype("int64").drop_duplicates().tolist())
    feitos = set(pd.read_csv(SAIDA).album_deezer_id.astype("int64")) if SAIDA.exists() else set()
    faltam = [i for i in ids if i not in feitos]
    print(f"{len(ids)} álbuns | faltam {len(faltam)}", flush=True)
    novo = not SAIDA.exists()
    with open(SAIDA, "a", newline="") as f, ThreadPoolExecutor(4) as ex:
        w = csv.DictWriter(f, ["album_deezer_id", "generos", "status"])
        if novo:
            w.writeheader()
        for n, fu in enumerate(as_completed([ex.submit(consultar, i) for i in faltam]), 1):
            if (linha := fu.result()):
                w.writerow(linha)
            if n % 500 == 0:
                f.flush(); print(f"{n}/{len(faltam)}", flush=True)
    print("fim", flush=True)


if __name__ == "__main__":
    main()
