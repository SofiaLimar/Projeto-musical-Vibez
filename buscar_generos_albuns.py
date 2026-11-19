"""
buscar_generos_albuns.py — gênero de cada álbum no Deezer.

Lê os album_deezer_id de data/popularidade.csv e grava data/generos_albuns.csv
(album_deezer_id, generos, status). "generos" vem separado por "|" (ex.: "Pop|Dance").
Retoma de onde parou.
"""
from pathlib import Path

import pandas as pd

import deezer

SAIDA = Path("data/generos_albuns.csv")


def generos(dados):
    nomes = [g.get("name", "") for g in (dados.get("genres") or {}).get("data", [])]
    return {"generos": "|".join(nomes)}


def main():
    ids = (pd.read_csv("data/popularidade.csv").album_deezer_id.dropna()
           .astype("int64").drop_duplicates().tolist())
    deezer.consultar_ids(ids, "/album/{}", SAIDA, ["album_deezer_id", "generos", "status"],
                         generos, nome="álbuns")


if __name__ == "__main__":
    main()
