"""
buscar_popularidade.py — popularidade das músicas no Deezer.

Para cada deezer_id com status "ok" em data/audio_teto100.csv, consulta /track/{id}
e grava o campo "rank" (popularidade do Deezer, 0 a ~1.000.000; maior = mais ouvida).

Saída: data/popularidade.csv (deezer_id, rank, artista_deezer_id, album_deezer_id, status)
Retoma de onde parou. Máximo 8 requisições/s (limite do Deezer: 50 a cada 5 s).

Uso:
  python buscar_popularidade.py --amostra 50    # teste
  python buscar_popularidade.py                 # todas (~25 min)
"""
import argparse, csv, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

DEEZER = "https://api.deezer.com"
INTERVALO = 0.125
SAIDA = Path("data/popularidade.csv")
CAMPOS = ["deezer_id", "rank", "artista_deezer_id", "album_deezer_id", "status"]

_trava, _ultimo = threading.Lock(), [0.0]
sessao = requests.Session()


def esperar_vez():
    with _trava:
        agora = time.monotonic()
        espera = _ultimo[0] + INTERVALO - agora
        if espera > 0:
            time.sleep(espera)
        _ultimo[0] = time.monotonic()


def consultar(deezer_id):
    for tentativa in range(6):
        esperar_vez()
        try:
            r = sessao.get(f"{DEEZER}/track/{deezer_id}", timeout=15)
            dados = r.json()
        except (requests.RequestException, ValueError):
            time.sleep(2 ** tentativa)
            continue
        erro = dados.get("error")
        if erro:
            if erro.get("code") == 4:              # limite de requisições: espera e tenta de novo
                time.sleep(5)
                continue
            return {"deezer_id": deezer_id, "rank": "", "artista_deezer_id": "", "album_deezer_id": "", "status": f"erro_{erro.get('code')}"}
        return {"deezer_id": deezer_id, "rank": dados.get("rank", ""),
                "artista_deezer_id": (dados.get("artist") or {}).get("id", ""),
                "album_deezer_id": (dados.get("album") or {}).get("id", ""), "status": "ok"}
    return None                                     # falha de conexão: não grava, tenta na próxima execução


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--amostra", type=int)
    p.add_argument("--trabalhadores", type=int, default=4)
    a = p.parse_args()

    idx = pd.read_csv("data/audio_teto100.csv")
    ids = idx.loc[idx.status == "ok", "deezer_id"].dropna().astype("int64").drop_duplicates().tolist()
    feitos = set()
    if SAIDA.exists():
        feitos = set(pd.read_csv(SAIDA).deezer_id.astype("int64"))
    faltam = [i for i in ids if i not in feitos]
    if a.amostra:
        faltam = faltam[:a.amostra]
    print(f"{len(ids)} músicas | já feitas {len(feitos)} | faltam {len(faltam)}", flush=True)

    novo = not SAIDA.exists()
    with open(SAIDA, "a", newline="") as f, ThreadPoolExecutor(a.trabalhadores) as ex:
        w = csv.DictWriter(f, CAMPOS)
        if novo:
            w.writeheader()
        futuros = [ex.submit(consultar, i) for i in faltam]
        for n, fu in enumerate(as_completed(futuros), 1):
            linha = fu.result()
            if linha:
                w.writerow(linha)
            if n % 500 == 0:
                f.flush()
                print(f"{n}/{len(faltam)}", flush=True)
    print("fim", flush=True)


if __name__ == "__main__":
    main()
