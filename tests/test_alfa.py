"""
Confere o α de Krippendorff de avaliacao/analisar.py com o exemplo publicado por Krippendorff
("Computing Krippendorff's Alpha-Reliability", 2011, seção C: 4 observadores, 12 unidades, com faltas).
Valores publicados: nominal 0,743 | ordinal 0,815 | intervalar 0,849.

Rodar na raiz do repositório:  python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "avaliacao"))
from analisar import alfa_krippendorff  # noqa: E402

N = None
OBSERVADORES = [
    [1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3],
    [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N],
]
UNIDADES = [[o[u] for o in OBSERVADORES if o[u] is not None] for u in range(12)]


class TestAlfa(unittest.TestCase):
    def test_exemplo_publicado(self):
        self.assertAlmostEqual(alfa_krippendorff(UNIDADES, "nominal"), 0.743, places=3)
        self.assertAlmostEqual(alfa_krippendorff(UNIDADES, "ordinal"), 0.815, places=3)
        self.assertAlmostEqual(alfa_krippendorff(UNIDADES, "intervalar"), 0.849, places=3)

    def test_concordancia_perfeita(self):
        self.assertAlmostEqual(alfa_krippendorff([[1, 1, 1], [3, 3, 3], [5, 5, 5]]), 1.0)


if __name__ == "__main__":
    unittest.main()
