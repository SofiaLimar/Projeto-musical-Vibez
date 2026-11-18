"""
Algoritmo do vibez: embedding CLIP de uma foto -> playlist.

Esta é a única cópia do algoritmo. Fica dentro de site/api/ para entrar no deploy
da Vercel; o "_" no nome impede a Vercel de tratá-la como um endpoint. Quem usa:
  - site/api/index.py (a API na Vercel)
  - foto_parte2_playlist.ipynb (seção 6)
  - tests/test_recomendar.py

Entra o embedding CLIP da foto (512 números, normalizado); sai a playlist.
O CLIP em si fica fora daqui: a API usa ONNX e o notebook usa PyTorch.

Dados (gerados pelo notebook, seção 5b, e por servidor/preparar_textos.py):
  modelo.json   Ridge como pesos W, b + media_prev/desvio_prev
  musicas.json  as músicas filtradas, na régua z
  textos.npz    embeddings de texto do CLIP (cenas de festa e rótulos do MuSe)
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

# ---------- parâmetros (README, seção 1) ----------
TAMANHO_PLAYLIST = 15
CANDIDATOS_POR_VAGA = 6
ALFA = 2          # peso da popularidade (varredura OASIS: 73% das músicas no top 10% de pop, dist p90 0,47)
MAX_POR_ARTISTA = 1
CANDIDATOS_GENERO = 30   # modo normal sem gênero escolhido: playlist no gênero dominante + 1 vizinho (15/11)

GEN_FESTA = {"Pop", "Dance", "Electro", "Rap/Hip Hop", "R&B"}   # gêneros do Deezer
LIMIAR_FESTA = 0.5
ALVO_FESTA = (0.5, 1.0)
BETA_FESTA = 0.5
TOP_ROTULOS = 15

# ---------- gênero (README, seção 2) ----------
GENEROS_SITE = ["Pop", "Rock", "Alternativo", "R&B", "Rap/Hip Hop", "Eletrônica", "Trilhas sonoras"]
VIZINHOS = {
    "Pop": ["R&B", "Alternativo"],
    "Rock": ["Alternativo", "Pop"],
    "Alternativo": ["Rock", "Pop"],
    "R&B": ["Pop", "Rap/Hip Hop"],
    "Rap/Hip Hop": ["R&B", "Pop"],
    "Eletrônica": ["Pop", "Alternativo"],
    "Trilhas sonoras": ["Alternativo", "Pop"],
}

PASTA_DADOS = Path(__file__).resolve().parent.parent / "_dados"   # site/api/_dados

# ---------- versão "clip" (Passo 4 do review, docs/REVIEW.md seção 5) ----------
# nota = cosseno(foto, rótulos da música) + λ · proximidade valência/energia + μ · rank(pop)
# Os três termos vão para a mesma escala (posição percentil entre as músicas, 0 a 1), como o review
# sugere para a popularidade; pesos iguais, definidos antes da avaliação (não ajustados nas fotos).
# Cosseno de uma música com vários rótulos = a média entre eles (o maior favoreceria quem tem mais rótulos).
# Sem modo festa e sem gênero dominante.
LAMBDA_CLIP = 1.0
MU_CLIP = 1.0
VERSOES = ("atual", "clip")


class SemMusicas(ValueError):
    """Nenhuma música atende aos gêneros escolhidos."""


def artista_base(nome):
    # "Bob Marley & The Wailers" e "Bob Marley" contam como o mesmo artista
    return re.split(r" & | feat\.? | ft\.? |, | and ", nome.lower())[0].strip()


def percentil(zv):
    return round(50 * (1 + math.erf(float(zv) / math.sqrt(2))), 1)


def posto_percentil(x):
    """Posição de cada valor entre todos (1/n = menor, 1 = maior); empates recebem a média das posições."""
    _, inv, cont = np.unique(x, return_inverse=True, return_counts=True)
    media = np.cumsum(cont) - (cont - 1) / 2
    return media[inv] / len(x)


def nome_playlist(z, festa):
    if festa:
        return "Clima de festa"
    alegre, agitada = z[0] >= 0, z[1] >= 0
    return {(True, True): "Alto astral", (True, False): "Leve e ensolarada",
            (False, True): "Intensa e inquieta", (False, False): "Calma e melancólica"}[(alegre, agitada)]


class Recomendador:
    def __init__(self, modelo, musicas, textos):
        self.W = np.array(modelo["W"], dtype=np.float32)            # (2, 512)
        self.B = np.array(modelo["b"], dtype=np.float32)
        self.media_prev = np.array(modelo["media_prev"], dtype=np.float32)
        self.desvio_prev = np.array(modelo["desvio_prev"], dtype=np.float32)

        self.musicas = musicas
        self.Zm = np.array([[m["zv"], m["ze"]] for m in musicas], dtype=np.float32)
        self.pop = np.array([m["pop"] for m in musicas], dtype=np.float32)
        self.gen_site = np.array([m["genero"] for m in musicas])
        self.gen_deezer = np.array([m["genero_deezer"] for m in musicas])
        self.artista = [artista_base(m["artista"]) for m in musicas]
        self.deezer_id = [m["deezer_id"] for m in musicas]

        self.t_cenas = textos["cenas"]
        self.n_festa = int(textos["n_festa"])
        self.t_rot = textos["rotulos"]
        self.rotulos = [str(r) for r in textos["nomes_rotulos"]]
        idx = {r: i for i, r in enumerate(self.rotulos)}
        self.listas_rot = [[idx[r] for r in m["rotulos"] if r in idx] for m in musicas]

    @classmethod
    def da_pasta(cls, pasta=PASTA_DADOS):
        pasta = Path(pasta)
        modelo = json.loads((pasta / "modelo.json").read_text())
        musicas = json.loads((pasta / "musicas.json").read_text())
        textos = dict(np.load(pasta / "textos.npz"))
        return cls(modelo, musicas, textos)

    # ---------- foto ----------
    def z_foto(self, e):
        p = self.W @ e + self.B
        return (p - self.media_prev) / self.desvio_prev

    def prob_festa(self, e):
        s = 100 * (self.t_cenas @ e)
        p = np.exp(s - s.max())
        p /= p.sum()
        return float(p[:self.n_festa].sum())

    def nota_rotulos(self, e):
        s = self.t_rot @ e
        top = np.argsort(-s)[:TOP_ROTULOS]
        w = np.zeros(len(self.rotulos))
        ex = np.exp(100 * (s[top] - s[top].max()))
        w[top] = ex / ex.max()
        tag = np.array([max((w[i] for i in ls), default=0.0) for ls in self.listas_rot])
        return tag, [self.rotulos[i] for i in top[:6]]

    # ---------- escolha ----------
    def escolher(self, nota, n=TAMANHO_PLAYLIST):
        escolhidas, por_artista, ids = [], {}, set()
        for i in np.argsort(nota):
            if not np.isfinite(nota[i]):
                break
            a = self.artista[i]
            if por_artista.get(a, 0) >= MAX_POR_ARTISTA or self.deezer_id[i] in ids:
                continue
            ids.add(self.deezer_id[i])
            por_artista[a] = por_artista.get(a, 0) + 1
            escolhidas.append(int(i))
            if len(escolhidas) == n:
                break
        return escolhidas

    def mesmo_quadrante(self, z):
        return (np.sign(self.Zm[:, 0]) == np.sign(z[0])) & (np.sign(self.Zm[:, 1]) == np.sign(z[1]))

    def generos_com_fallback(self, generos, z, n=TAMANHO_PLAYLIST):
        """Se os gêneros escolhidos têm < n músicas no quadrante da foto, completa com vizinhos."""
        usados = list(generos)
        quad = self.mesmo_quadrante(z)
        acrescentados = []
        for viz in [v for g in generos for v in VIZINHOS[g]]:
            if (np.isin(self.gen_site, usados) & quad).sum() >= n:
                break
            if viz not in usados:
                usados.append(viz)
                acrescentados.append(viz)
        aviso = None
        if acrescentados:
            aviso = (f"Poucas músicas de {' e '.join(generos)} combinam com o clima desta foto; "
                     f"completamos com {' e '.join(acrescentados)}.")
        return usados, aviso

    def playlist(self, e, generos=(), versao="atual"):
        """e: embedding CLIP da foto (512, normalizado). generos: nomes de GENEROS_SITE.
        versao: "atual" (a que está no ar) ou "clip" (Passo 4 do review)."""
        if versao == "clip":
            return self.playlist_clip(e, generos)
        if versao != "atual":
            raise ValueError(f"versão desconhecida: {versao}")
        z = self.z_foto(e)
        pf = self.prob_festa(e)
        festa = pf >= LIMIAR_FESTA
        tag, rotulos = self.nota_rotulos(e)
        aviso = None
        n = TAMANHO_PLAYLIST
        Zm, pop, gen_site = self.Zm, self.pop, self.gen_site

        if festa:
            alvo = np.array([max(z[0], ALVO_FESTA[0]), max(z[1], ALVO_FESTA[1])])
            gen_ok = np.isin(gen_site, generos) if generos else np.isin(self.gen_deezer, list(GEN_FESTA))
            ok = gen_ok & (Zm[:, 0] > 0) & (Zm[:, 1] > 0.3)
            if ok.sum() < n * 2:
                ok = gen_ok
            dist = np.linalg.norm(Zm - alvo, axis=1)
            nota = np.where(ok, dist - ALFA * pop - BETA_FESTA * tag, np.inf)
        else:
            if generos:
                usados, aviso = self.generos_com_fallback(generos, z)
                ok = np.isin(gen_site, usados)
            else:
                ok = np.ones(len(self.musicas), bool)
            dist = np.linalg.norm(Zm - z, axis=1)

            def nota_perto(ok, k):
                k = min(int(ok.sum()), k)
                perto = np.argsort(np.where(ok, dist, np.inf))[:k]
                nota = np.full(len(self.musicas), np.inf)
                nota[perto] = dist[perto] - ALFA * pop[perto]
                return nota

            nota = nota_perto(ok, n * CANDIDATOS_POR_VAGA)   # as 90 mais próximas
            if not generos:
                # sem gênero escolhido: usa o gênero mais comum da playlist provisória + 1 vizinho,
                # para não misturar reggae, k-pop e emo quando a foto cai no centro do mapa
                contagem = Counter(g for g in gen_site[self.escolher(nota)] if g in VIZINHOS)
                if contagem:
                    dom = contagem.most_common(1)[0][0]
                    nota = nota_perto(np.isin(gen_site, [dom, VIZINHOS[dom][0]]), CANDIDATOS_GENERO)

        escolhidas = self.escolher(nota)
        if not escolhidas:
            raise SemMusicas("Não há músicas suficientes nesses gêneros.")
        return self._resposta(z, festa, pf, rotulos, aviso, escolhidas)

    def cosseno_rotulos(self, e):
        """Para cada música, a média dos cossenos entre a foto e os seus rótulos do MuSe."""
        s = self.t_rot @ e   # os embeddings de texto e da foto já são normalizados: produto = cosseno
        cos = np.array([s[ls].mean() if ls else -1.0 for ls in self.listas_rot])
        return cos, [self.rotulos[i] for i in np.argsort(-s)[:6]]

    def playlist_clip(self, e, generos=()):
        z = self.z_foto(e)
        pf = self.prob_festa(e)   # só informativo: esta versão não tem modo festa
        cos, rotulos = self.cosseno_rotulos(e)
        ok = np.isin(self.gen_site, generos) if generos else np.ones(len(self.musicas), bool)
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            raise SemMusicas("Não há músicas suficientes nesses gêneros.")
        dist = np.linalg.norm(self.Zm[idx] - z, axis=1)
        nota = (posto_percentil(cos[idx])
                + LAMBDA_CLIP * posto_percentil(-dist)          # mais perto = posição maior
                + MU_CLIP * posto_percentil(self.pop[idx]))
        completa = np.full(len(self.musicas), np.inf)
        completa[idx] = -nota                                   # escolher() pega as menores
        escolhidas = self.escolher(completa)
        if not escolhidas:
            raise SemMusicas("Não há músicas suficientes nesses gêneros.")
        return self._resposta(z, False, pf, rotulos, None, escolhidas)

    def _resposta(self, z, festa, pf, rotulos, aviso, escolhidas):
        return {
            "nome": nome_playlist(z, festa),
            "valencia_pct": percentil(z[0]),
            "energia_pct": percentil(z[1]),
            "z": [round(float(z[0]), 3), round(float(z[1]), 3)],
            "modo_festa": bool(festa),
            "prob_festa": round(pf, 3),
            "rotulos": rotulos,
            "aviso": aviso,
            "link_playlist": None,
            "faixas": [{
                "titulo": self.musicas[i]["titulo"],
                "artista": self.musicas[i]["artista"],
                "genero": self.musicas[i]["genero"] or self.musicas[i]["genero_deezer"],
                "deezer_id": self.musicas[i]["deezer_id"],
                "previa_url": None,   # o site busca a prévia no Deezer na hora
            } for i in escolhidas],
        }
