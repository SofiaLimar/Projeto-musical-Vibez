"""
Analisa as respostas da avaliação humana (docs/avaliacao.md, seção 5).

Uso (na raiz do repositório):
  python avaliacao/analisar.py --chave avaliacao/saida/chave.json --respostas avaliacao/respostas
(--respostas: pasta com os .csv exportados pela página, um por pessoa)
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def alfa_krippendorff(unidades, metrica="ordinal"):
    """
    α de Krippendorff pela matriz de coincidências (Krippendorff, 2004; 2011).
    unidades: lista de listas com os valores dados a cada unidade (só as notas existentes).
    Unidades com menos de 2 valores não entram (não há par para comparar).
    """
    unidades = [u for u in unidades if len(u) >= 2]
    valores = sorted({v for u in unidades for v in u})
    if len(valores) < 2:
        return float("nan")
    pos = {v: i for i, v in enumerate(valores)}
    o = np.zeros((len(valores), len(valores)))
    for u in unidades:
        m = len(u)
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[pos[u[a]], pos[u[b]]] += 1 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    v = np.array(valores, dtype=float)
    if metrica == "nominal":
        d2 = (v[:, None] != v[None, :]).astype(float)
    elif metrica == "intervalar":
        d2 = (v[:, None] - v[None, :]) ** 2
    elif metrica == "ordinal":
        acum = np.cumsum(n_c)
        d2 = np.zeros_like(o)
        for c in range(len(v)):
            for k in range(len(v)):
                lo, hi = min(c, k), max(c, k)
                soma = acum[hi] - (acum[lo - 1] if lo > 0 else 0)
                d2[c, k] = (soma - (n_c[c] + n_c[k]) / 2) ** 2
    else:
        raise ValueError(metrica)
    observado = (o * d2).sum()
    esperado = (np.outer(n_c, n_c) * d2).sum() / (n - 1)
    return 1 - observado / esperado


def ler_respostas(pasta, estudo):
    notas = {}   # (avaliador, codigo) -> (nota, previas)
    for arq in sorted(Path(pasta).glob("*.csv")):
        with open(arq, encoding="utf-8-sig", newline="") as f:
            for linha in csv.DictReader(f):
                if linha["estudo"] != estudo:
                    sys.exit(f"{arq.name} é de outro estudo ({linha['estudo']}); esperado {estudo}.")
                notas[(linha["avaliador"], linha["codigo"])] = (int(linha["nota"]), int(linha["previas_ouvidas"]))
    return notas


def bootstrap_ic(x, n=10_000, semente=0):
    rng = np.random.default_rng(semente)
    medias = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.quantile(medias, [0.025, 0.975])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chave", required=True, type=Path)
    ap.add_argument("--respostas", required=True, type=Path)
    args = ap.parse_args()

    chave = json.loads(args.chave.read_text(encoding="utf-8"))
    codigos = chave["codigos"]
    notas = ler_respostas(args.respostas, chave["estudo"])
    avaliadores = sorted({a for a, _ in notas})
    versoes = list(chave["versoes"])
    print(f"estudo {chave['estudo']} | {len(avaliadores)} pessoa(s): {', '.join(avaliadores)} | versões: {', '.join(versoes)}")

    desconhecidos = {c for _, c in notas if c not in codigos}
    if desconhecidos:
        sys.exit(f"códigos que não estão na chave: {sorted(desconhecidos)[:5]}")

    # 1. concordância: cada playlist (código) é uma unidade
    por_codigo = defaultdict(list)
    for (a, c), (nota, _) in notas.items():
        por_codigo[c].append(nota)
    unidades = list(por_codigo.values())
    print("\n1. Concordância entre as pessoas (α de Krippendorff)")
    if len(avaliadores) < 2:
        print("   precisa de pelo menos 2 pessoas")
    else:
        a_ord = alfa_krippendorff(unidades, "ordinal")
        leitura = ("confiável" if a_ord >= 0.8 else "só para conclusões provisórias" if a_ord >= 0.667 else "não conclusivo")
        print(f"   ordinal {a_ord:.3f} ({leitura}) | intervalar {alfa_krippendorff(unidades, 'intervalar'):.3f} "
              f"| playlists com 2+ notas: {sum(len(u) >= 2 for u in unidades)}")

    # 2. nota de cada versão
    print("\n2. Nota média de cada versão")
    media_foto = defaultdict(dict)   # foto -> versao -> média das pessoas
    for c, ns in por_codigo.items():
        media_foto[codigos[c]["foto"]][codigos[c]["versao"]] = float(np.mean(ns))
    for v in versoes:
        todas = [n for (a, c), (n, _) in notas.items() if codigos[c]["versao"] == v]
        if todas:
            print(f"   {v}: {np.mean(todas):.2f} (desvio {np.std(todas, ddof=1) if len(todas) > 1 else 0:.2f}, {len(todas)} notas)")

    # 3. comparação pareada (Passo 4)
    if len(versoes) == 2:
        a, b = versoes
        fotos = sorted(f for f, d in media_foto.items() if a in d and b in d)
        dif = np.array([media_foto[f][b] - media_foto[f][a] for f in fotos])
        print(f"\n3. Comparação {b} − {a} ({len(fotos)} fotos com as duas versões)")
        if len(fotos) >= 2:
            from scipy.stats import wilcoxon
            lo, hi = bootstrap_ic(dif)
            p = wilcoxon(dif).pvalue if np.any(dif != 0) else 1.0
            print(f"   diferença média {dif.mean():+.2f} | IC 95% (bootstrap) [{lo:+.2f}, {hi:+.2f}] | Wilcoxon p = {p:.3f}")
            print(f"   {b} melhor em {int((dif > 0).sum())} fotos, pior em {int((dif < 0).sum())}, empate em {int((dif == 0).sum())}")

    # 4. esforço
    print("\n4. Prévias ouvidas por playlist (mínimo pedido: 3)")
    for av in avaliadores:
        ps = [p for (a, _), (_, p) in notas.items() if a == av]
        print(f"   {av}: mediana {np.median(ps):.0f} | playlists com menos de 3: {sum(p < 3 for p in ps)} de {len(ps)}")


if __name__ == "__main__":
    main()
