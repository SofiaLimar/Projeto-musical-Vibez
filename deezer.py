"""
deezer.py — cliente da API do Deezer usado pelos scripts de coleta
(baixar_audio.py, buscar_popularidade.py, buscar_fas_artistas.py, buscar_generos_albuns.py).

- controle de velocidade compartilhado por todas as threads: no máximo 8 requisições/s
  (limite do Deezer: 50 a cada 5 s);
- tenta de novo quando a conexão falha e quando o Deezer pede para esperar (erro 4);
- consultar_ids(): consulta um endpoint para uma lista de ids e grava um CSV, retomando de onde parou.
"""
import csv
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

DEEZER = "https://api.deezer.com"
INTERVALO = 0.125               # 8 requisições/s no total, somando todos os trabalhadores

_trava = threading.Lock()
_ultimo = [0.0]
avisos_limite = [0]             # quantas vezes o Deezer pediu para esperar (erro 4)
sessao = requests.Session()


def esperar_vez():
    """Garante o intervalo mínimo entre requisições à API, somando todos os trabalhadores."""
    with _trava:
        espera = _ultimo[0] + INTERVALO - time.monotonic()
        if espera > 0:
            time.sleep(espera)
        _ultimo[0] = time.monotonic()


def pedir(caminho, params=None, tentativas=6, timeout=15):
    """
    GET na API. `caminho` é "/track/123" ou uma URL completa.
    Devolve o JSON; se o Deezer responder com erro (exceto o 4, que é tentado de novo),
    devolve esse JSON com a chave "error". Devolve None se a conexão falhar em todas as tentativas.
    """
    url = caminho if caminho.startswith("http") else DEEZER + caminho
    for tentativa in range(tentativas):
        esperar_vez()
        try:
            r = sessao.get(url, params=params, timeout=timeout)
            if r.status_code != 200:
                raise requests.RequestException(f"HTTP {r.status_code}")
            dados = r.json()
        except (requests.RequestException, ValueError):
            time.sleep(2 ** tentativa)
            continue
        if isinstance(dados, dict) and (dados.get("error") or {}).get("code") == 4:
            with _trava:
                avisos_limite[0] += 1
            time.sleep(5 * (tentativa + 1))
            continue
        return dados
    return None


def consultar_ids(ids, endpoint, saida, campos, extrair, trabalhadores=4, amostra=None, nome="itens"):
    """
    Para cada id, pede endpoint.format(id) e grava uma linha em `saida` (CSV), retomando de onde parou.
    campos: colunas do CSV; a 1ª é o id e a última é "status".
    extrair(dados) -> dict com as colunas do meio.
    Erro do Deezer vira status "erro_<código>"; falha de conexão não é gravada (é tentada na próxima execução).
    """
    chave = campos[0]
    feitos = set(pd.read_csv(saida)[chave].astype("int64")) if saida.exists() else set()
    faltam = [i for i in ids if i not in feitos]
    if amostra:
        faltam = faltam[:amostra]
    print(f"{len(ids)} {nome} | já feitos {len(feitos)} | faltam {len(faltam)}", flush=True)

    def consultar(i):
        dados = pedir(endpoint.format(i))
        if dados is None:
            return None
        erro = dados.get("error")
        if erro:
            return {chave: i, **{c: "" for c in campos[1:-1]}, "status": f"erro_{erro.get('code')}"}
        return {chave: i, **extrair(dados), "status": "ok"}

    novo = not saida.exists()
    with open(saida, "a", newline="") as f, ThreadPoolExecutor(trabalhadores) as ex:
        w = csv.DictWriter(f, campos)
        if novo:
            w.writeheader()
        for n, fu in enumerate(as_completed([ex.submit(consultar, i) for i in faltam]), 1):
            if (linha := fu.result()):
                w.writerow(linha)
            if n % 500 == 0:
                f.flush()
                print(f"{n}/{len(faltam)}", flush=True)
    print("fim", flush=True)
