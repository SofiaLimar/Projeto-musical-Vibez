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


class TestEngenharia(unittest.TestCase):
    def test_mesmo_quadrante_conta_zero_como_positivo(self):
        # foto exatamente em 0: casa com as músicas de valência/energia >= 0, como em nome_playlist
        ok = rec.mesmo_quadrante(np.array([0.0, 0.0]))
        esperado = (rec.Zm[:, 0] >= 0) & (rec.Zm[:, 1] >= 0)
        self.assertTrue(ok.any())
        np.testing.assert_array_equal(ok, esperado)

    def test_top_rotulos(self):
        e = embedding_fixo(5)
        s = rec.t_rot @ e
        self.assertEqual(rec.top_rotulos(e), [rec.rotulos[i] for i in np.argsort(-s)[:6]])


class TestVersaoClip(unittest.TestCase):
    """Versão do Passo 4: cosseno com os rótulos + proximidade + popularidade, na mesma escala."""

    def conferir_faixas(self, r):
        TestPlaylist.conferir_faixas(self, r)

    def test_da_15_faixas_sem_repetir_artista(self):
        for semente in range(5):
            with self.subTest(semente=semente):
                self.conferir_faixas(rec.playlist(embedding_fixo(semente), versao="clip"))

    def test_nao_tem_modo_festa(self):
        r = rec.playlist(embedding_de_festa(), versao="clip")
        self.assertFalse(r["modo_festa"])
        self.conferir_faixas(r)

    def test_mesma_foto_da_mesma_playlist(self):
        self.assertEqual(rec.playlist(embedding_fixo(), versao="clip"),
                         rec.playlist(embedding_fixo(), versao="clip"))

    def test_genero_escolhido(self):
        for g in GENEROS_SITE:
            with self.subTest(genero=g):
                r = rec.playlist(embedding_fixo(), [g], versao="clip")
                self.assertTrue({f["genero"] for f in r["faixas"]} <= {g})

    def test_formula(self):
        # a 1ª música escolhida tem a maior nota da fórmula entre todas
        from _vibez.recomendar import LAMBDA_CLIP, MU_CLIP, posto_percentil
        e = embedding_fixo(3)
        cos, _ = rec.cosseno_rotulos(e)
        dist = np.linalg.norm(rec.Zm - rec.z_foto(e), axis=1)
        nota = posto_percentil(cos) + LAMBDA_CLIP * posto_percentil(-dist) + MU_CLIP * posto_percentil(rec.pop)
        primeira = rec.playlist(e, versao="clip")["faixas"][0]["deezer_id"]
        self.assertEqual(primeira, rec.deezer_id[int(np.argmax(nota))])

    def test_cosseno_e_a_media_dos_rotulos(self):
        e = embedding_fixo(4)
        cos, _ = rec.cosseno_rotulos(e)
        s = rec.t_rot @ e
        i = next(k for k, ls in enumerate(rec.listas_rot) if len(ls) >= 2)
        self.assertAlmostEqual(cos[i], float(np.mean([s[j] for j in rec.listas_rot[i]])), places=6)

    def test_posto_percentil(self):
        from _vibez.recomendar import posto_percentil
        np.testing.assert_allclose(posto_percentil(np.array([10, 30, 20, 20])), [0.25, 1.0, 0.625, 0.625])

    def test_versao_desconhecida(self):
        with self.assertRaises(ValueError):
            rec.playlist(embedding_fixo(), versao="outra")


if __name__ == "__main__":
    unittest.main()
