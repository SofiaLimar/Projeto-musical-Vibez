"""
Testes do algoritmo (site/api/_vibez/recomendar.py), com os dados reais de site/api/_dados.
Não precisam do CLIP: usam embeddings fixos.

Rodar na raiz do repositório:  python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "site" / "api"))
from _vibez.recomendar import (GENEROS_SITE, TAMANHO_PLAYLIST, VIZINHOS,  # noqa: E402
                              Recomendador, artista_base)

rec = Recomendador.da_pasta()


def embedding_fixo(semente=0):
    v = np.random.default_rng(semente).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def embedding_de_festa():
    # embedding de texto do CLIP para "a photo of a party": cai no modo festa
    return rec.t_cenas[0].astype(np.float32)


class TestPlaylist(unittest.TestCase):
    def conferir_faixas(self, r):
        faixas = r["faixas"]
        self.assertEqual(len(faixas), TAMANHO_PLAYLIST)
        artistas = [artista_base(f["artista"]) for f in faixas]
        self.assertEqual(len(set(artistas)), len(artistas), "artista repetido")
        ids = [f["deezer_id"] for f in faixas]
        self.assertEqual(len(set(ids)), len(ids), "música repetida")

    def test_embedding_fixo_da_15_faixas_sem_repetir_artista(self):
        r = rec.playlist(embedding_fixo())
        self.assertFalse(r["modo_festa"])
        self.conferir_faixas(r)

    def test_mesma_foto_da_mesma_playlist(self):
        a = rec.playlist(embedding_fixo())
        b = rec.playlist(embedding_fixo())
        self.assertEqual(a, b)

    def test_foto_de_festa_ativa_o_modo_festa(self):
        r = rec.playlist(embedding_de_festa())
        self.assertTrue(r["modo_festa"])
        self.assertGreaterEqual(r["prob_festa"], 0.5)
        self.assertEqual(r["nome"], "Clima de festa")
        self.conferir_faixas(r)

    def test_genero_escolhido_respeita_genero_e_vizinhos(self):
        for g in GENEROS_SITE:
            with self.subTest(genero=g):
                r = rec.playlist(embedding_fixo(), [g])
                self.conferir_faixas(r)
                permitidos = {g, *VIZINHOS[g]}
                self.assertTrue({f["genero"] for f in r["faixas"]} <= permitidos)


if __name__ == "__main__":
    unittest.main()
