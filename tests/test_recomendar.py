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

    def test_percentil_empirico(self):
        from _vibez.recomendar import percentil
        ref = np.array([-1.0, 0.0, 1.0, 2.0])
        self.assertEqual(percentil(0.0, ref), 37.5)      # 1 abaixo, 1 empate (conta metade), de 4
        self.assertEqual(percentil(5.0, ref), 100.0)
        self.assertEqual(percentil(-5.0, ref), 0.0)
        self.assertEqual(percentil(0.0), 50.0)           # sem referência: curva normal

    def test_top_rotulos(self):
        e = embedding_fixo(5)
        s = rec.t_rot @ e
        self.assertEqual(rec.top_rotulos(e), [rec.rotulos[i] for i in np.argsort(-s)[:6]])


class TestVersaoMahalanobis(unittest.TestCase):
    """Candidata do review (linha 48): igual à atual, mas com distância de Mahalanobis."""

    def conferir_faixas(self, r):
        TestPlaylist.conferir_faixas(self, r)

    def test_da_15_faixas_sem_repetir_artista(self):
        for semente in range(5):
            with self.subTest(semente=semente):
                self.conferir_faixas(rec.playlist(embedding_fixo(semente), versao="mahalanobis"))

    def test_mantem_o_modo_festa(self):
        r = rec.playlist(embedding_de_festa(), versao="mahalanobis")
        self.assertTrue(r["modo_festa"])
        self.conferir_faixas(r)

    def test_genero_escolhido(self):
        for g in GENEROS_SITE:
            with self.subTest(genero=g):
                r = rec.playlist(embedding_fixo(), [g], versao="mahalanobis")
                self.assertTrue({f["genero"] for f in r["faixas"]} <= {g, *VIZINHOS[g]})

    def test_distancia(self):
        alvo = np.array([0.3, -0.2], dtype=np.float32)
        d = rec.Zm - alvo
        esperado = np.sqrt([v @ rec.S_inv @ v for v in d[:50]])
        np.testing.assert_allclose(rec.distancia(alvo, True)[:50], esperado, rtol=1e-5)
        np.testing.assert_allclose(rec.distancia(alvo, False), np.linalg.norm(d, axis=1), rtol=1e-6)

    def test_formula(self):
        # com gênero escolhido (sem a etapa do gênero dominante): a 1ª faixa tem a menor nota
        # (distância de Mahalanobis − ALFA · pop) entre as 90 mais próximas por Mahalanobis
        from _vibez.recomendar import ALFA
        e = embedding_fixo(2)
        z = rec.z_foto(e)
        usados, _ = rec.generos_com_fallback(["Pop"], z)
        ok = np.isin(rec.gen_site, usados)
        dist = rec.distancia(z, True)
        perto = np.argsort(np.where(ok, dist, np.inf))[:90]
        nota = dist[perto] - ALFA * rec.pop[perto]
        primeira = rec.playlist(e, ["Pop"], versao="mahalanobis")["faixas"][0]["deezer_id"]
        self.assertEqual(primeira, rec.deezer_id[int(perto[np.argmin(nota)])])

    def test_muda_a_playlist(self):
        difere = [rec.playlist(embedding_fixo(k))["faixas"] != rec.playlist(embedding_fixo(k), versao="mahalanobis")["faixas"]
                  for k in range(20)]
        self.assertTrue(any(difere))


class TestVersaoDanca(unittest.TestCase):
    """Candidata do review (linha 65): modo festa filtrado pela dançabilidade do áudio."""

    def conferir_faixas(self, r):
        TestPlaylist.conferir_faixas(self, r)

    def test_todas_as_musicas_tem_dancabilidade(self):
        self.assertFalse(np.isnan(rec.danca).any())

    def test_festa_so_com_musicas_dancantes(self):
        from _vibez.recomendar import LIMIAR_DANCA
        r = rec.playlist(embedding_de_festa(), versao="danca")
        self.assertTrue(r["modo_festa"])
        self.conferir_faixas(r)
        for f in r["faixas"]:
            self.assertGreater(rec.danca[rec.deezer_id.index(f["deezer_id"])], LIMIAR_DANCA)

    def test_festa_com_rock_muda_a_playlist(self):
        # sem gênero escolhido, as playlists de festa quase não mudam (o alvo já puxa para alegre e agitada);
        # a diferença aparece quando a pessoa escolhe um gênero, como Rock (medido em docs/decisoes.md)
        e = embedding_de_festa()
        self.assertNotEqual(rec.playlist(e, ["Rock"])["faixas"], rec.playlist(e, ["Rock"], versao="danca")["faixas"])

    def test_fora_da_festa_igual_a_atual(self):
        for k in range(10):
            e = embedding_fixo(k)
            atual = rec.playlist(e)
            if not atual["modo_festa"]:
                self.assertEqual(atual["faixas"], rec.playlist(e, versao="danca")["faixas"])


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
