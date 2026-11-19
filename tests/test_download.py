"""
Testes do download do modelo (site/api/_clip_onnx.py, baixar), sem rede: um servidor falso entrega os bytes.

Rodar na raiz do repositório:  python -m unittest discover tests
"""
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "site" / "api"))
from _clip_onnx import baixar  # noqa: E402

CONTEUDO = b"modelo" * 1_000_000          # ~6 MB, em vários blocos de 4 MB
SHA = hashlib.sha256(CONTEUDO).hexdigest()


def abrir_falso(conteudo, atraso=0.0):
    import time

    class Resposta(io.BytesIO):
        def read(self, n=-1):
            time.sleep(atraso)
            return super().read(n)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    return lambda url, timeout: Resposta(conteudo)


class TestBaixar(unittest.TestCase):
    def setUp(self):
        self.pasta = Path(tempfile.mkdtemp())
        self.destino = self.pasta / "modelo.onnx"

    def test_download_certo(self):
        baixar("x", self.destino, len(CONTEUDO), SHA, 30, abrir_falso(CONTEUDO))
        self.assertEqual(self.destino.read_bytes(), CONTEUDO)

    def test_sha256_errado_nao_grava(self):
        estragado = CONTEUDO[:-1] + b"X"
        with self.assertRaisesRegex(RuntimeError, "sha256"):
            baixar("x", self.destino, len(CONTEUDO), SHA, 30, abrir_falso(estragado))
        self.assertEqual(list(self.pasta.iterdir()), [])

    def test_incompleto_nao_grava(self):
        with self.assertRaisesRegex(RuntimeError, "incompleto"):
            baixar("x", self.destino, len(CONTEUDO), SHA, 30, abrir_falso(CONTEUDO[:-10]))
        self.assertEqual(list(self.pasta.iterdir()), [])

    def test_passou_do_prazo(self):
        with self.assertRaisesRegex(RuntimeError, "passou de"):
            baixar("x", self.destino, len(CONTEUDO), SHA, 0.05, abrir_falso(CONTEUDO, atraso=0.04))
        self.assertEqual(list(self.pasta.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
