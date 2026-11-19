"""
buscar_popularidade.py — popularidade das músicas no Deezer.

Para cada deezer_id com status "ok" em data/audio_teto100.csv, consulta /track/{id}
e grava o campo "rank" (popularidade do Deezer, 0 a ~1.000.000; maior = mais ouvida).

Saída: data/popularidade.csv (deezer_id, rank, artista_deezer_id, album_deezer_id, status)
Retoma de onde parou. Máximo 8 requisições/s (limite do Deezer: 50 a cada 5 s; ver deezer.py).

Uso:
  python buscar_popularidade.py --amostra 50    # teste
  python buscar_popularidade.py                 # todas (~25 min)
"""
import argparse
from pathlib import Path

import pandas as pd

import deezer

SAIDA = Path("data/popularidade.csv")
CAMPOS = ["deezer_id", "rank", "artista_deezer_id", "album_deezer_id", "status"]


def popularidade(dados):
    return {"rank": dados.get("rank", ""),
            "artista_deezer_id": (dados.get("artist") or {}).get("id", ""),
            "album_deezer_id": (dados.get("album") or {}).get("id", "")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--amostra", type=int)
    p.add_argument("--trabalhadores", type=int, default=4)
    a = p.parse_args()

    idx = pd.read_csv("data/audio_teto100.csv")
    ids = idx.loc[idx.status == "ok", "deezer_id"].dropna().astype("int64").drop_duplicates().tolist()
    deezer.consultar_ids(ids, "/track/{}", SAIDA, CAMPOS, popularidade,
                         trabalhadores=a.trabalhadores, amostra=a.amostra, nome="músicas")


if __name__ == "__main__":
    main()
