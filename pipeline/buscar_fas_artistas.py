"""
buscar_fas_artistas.py — número de fãs de cada artista no Deezer (nb_fan).

Lê os artista_deezer_id de data/popularidade.csv (de buscar_popularidade.py)
e grava data/fas_artistas.csv (artista_deezer_id, nb_fan, status). Retoma de onde parou.
"""
from pathlib import Path

import pandas as pd

import deezer

SAIDA = Path("data/fas_artistas.csv")


def main():
    ids = (pd.read_csv("data/popularidade.csv").artista_deezer_id.dropna()
           .astype("int64").drop_duplicates().tolist())
    deezer.consultar_ids(ids, "/artist/{}", SAIDA, ["artista_deezer_id", "nb_fan", "status"],
                         lambda d: {"nb_fan": d.get("nb_fan", "")}, nome="artistas")


if __name__ == "__main__":
    main()
