"""
buscar_fas_artistas.py — número de fãs de cada artista no Deezer (nb_fan).

Lê os artista_deezer_id de data/popularidade.csv (de buscar_popularidade.py)
e grava data/fas_artistas.csv (artista_deezer_id, nb_fan, status). Retoma de onde parou.
"""
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from buscar_popularidade import esperar_vez, sessao, DEEZER
import time, requests

SAIDA = Path("data/fas_artistas.csv")


def consultar(art_id):
    for tentativa in range(6):
        esperar_vez()
        try:
            dados = sessao.get(f"{DEEZER}/artist/{art_id}", timeout=15).json()
        except (requests.RequestException, ValueError):
            time.sleep(2 ** tentativa)
            continue
        erro = dados.get("error")
        if erro:
            if erro.get("code") == 4:
                time.sleep(5)
                continue
            return {"artista_deezer_id": art_id, "nb_fan": "", "status": f"erro_{erro.get('code')}"}
        return {"artista_deezer_id": art_id, "nb_fan": dados.get("nb_fan", ""), "status": "ok"}
    return None


def main():
    ids = (pd.read_csv("data/popularidade.csv").artista_deezer_id.dropna()
           .astype("int64").drop_duplicates().tolist())
    feitos = set(pd.read_csv(SAIDA).artista_deezer_id.astype("int64")) if SAIDA.exists() else set()
    faltam = [i for i in ids if i not in feitos]
    print(f"{len(ids)} artistas | faltam {len(faltam)}", flush=True)
    novo = not SAIDA.exists()
    with open(SAIDA, "a", newline="") as f, ThreadPoolExecutor(4) as ex:
        w = csv.DictWriter(f, ["artista_deezer_id", "nb_fan", "status"])
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
