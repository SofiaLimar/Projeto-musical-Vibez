# vibez: playlist a partir da vibe de uma foto

A pessoa envia uma foto (paisagem, cenário, retrato…) e opcionalmente escolhe gêneros. O sistema dá à foto uma nota de **humor (valência)** e de **energia (arousal)** e devolve 15 músicas conhecidas com o mesmo clima.

No ar em **https://vibez-azure.vercel.app**.

- **Como funciona e por quê** (algoritmo, gêneros, dados, contrato da API, números medidos): [`docs/decisoes.md`](docs/decisoes.md)
- **Avaliação com pessoas** (protocolo e resultado): [`docs/avaliacao.md`](docs/avaliacao.md)
- **Review do repositório** (o que foi apontado e a ordem dos passos): [`docs/REVIEW.md`](docs/REVIEW.md)
- **Dados** (de onde vem cada arquivo, licenças, o que é final e o que é intermediário): [`data/README.md`](data/README.md)

## Como funciona, em uma frase

O CLIP transforma a foto em 512 números, um Ridge treinado no OASIS tira deles humor e energia, e a API escolhe as 15 músicas mais próximas nesse mapa entre 2.812 músicas (com peso para popularidade, no máximo uma por artista e um modo festa). Detalhes em `docs/decisoes.md`, seção 1.

## Como rodar

Comandos na raiz do repositório.

**Testes** (só precisam de numpy e Pillow):
```
python -m unittest discover tests
```

**Site + API no computador** (precisa de `pip install -r site/requirements.txt` e `uvicorn`):
```
cd ferramentas
CLIP_ONNX=<caminho do vision_model.onnx> uvicorn rodar_local:app --port 8010
```
Sem `CLIP_ONNX`, o modelo é baixado da Hugging Face na primeira foto.

**Publicar:** um push na `main` faz a Vercel publicar a pasta `site/`.

**Refazer os dados** (dependências em `requirements-pipeline.txt`), nesta ordem:

| Passo | Onde | Gera |
|---|---|---|
| 1. `pipeline/contar_rotulos.py` (precisa de `LASTFM_API_KEY`) | Mac | `data/lista_rotulos_1000.csv` |
| 2. `pipeline/baixar_audio.py` | Mac | `audio_previas/` e `data/audio_teto100.csv` |
| 3. `notebooks/notas_essentia.ipynb` | Colab | `notas_musicas.csv` (valência/energia de cada prévia) |
| 4. `pipeline/buscar_popularidade.py`, `buscar_fas_artistas.py`, `buscar_generos_albuns.py`, `montar_popularidade.py` | Mac | `data/popularidade_final.csv` |
| 5. `notebooks/foto_parte1_treino.ipynb` | Colab | o modelo de foto (`foto_humor_energia.joblib`) |
| 6. `notebooks/foto_parte2_playlist.ipynb`, seção 5b | Colab | `site/api/_dados/modelo.json` e `musicas.json` |
| 7. `ferramentas/preparar_textos.py` | Mac | `site/api/_dados/textos.npz` |

Os notebooks rodam no Colab e leem/gravam numa pasta do Google Drive (a primeira célula de cada um diz qual); veja `data/README.md`.

## Arquivos

| Arquivo | Função |
|---|---|
| `notebooks/foto_parte1_treino.ipynb` | Colab: treina CLIP + Ridge no OASIS e salva o `.joblib` |
| `notebooks/notas_essentia.ipynb` | Colab: notas de valência/energia das prévias (MusiCNN + emoMusic) → `notas_musicas.csv` |
| `notebooks/foto_parte2_playlist.ipynb` | Colab: limpeza, popularidade, gênero, régua e playlist (a playlist usa `site/api/_vibez/recomendar.py`) |
| `notebooks/validar_onnx_quantizado.ipynb` | Colab: valida o CLIP quantizado (89 MB) contra o PyTorch antes de ligá-lo no site |
| `data/` | tabelas do pipeline; origem, licença e o que é final em `data/README.md` |
| `pipeline/buscar_popularidade.py`, `buscar_fas_artistas.py`, `buscar_generos_albuns.py` | coletam no Deezer o rank, os fãs e os gêneros |
| `pipeline/deezer.py` | cliente do Deezer usado pelos scripts de coleta e pelo `baixar_audio.py` (máx. 8 req/s, novas tentativas, retomada) |
| `pipeline/montar_popularidade.py` | gera `data/popularidade_final.csv` |
| `pipeline/contar_rotulos.py`, `baixar_audio.py` | montaram a base (Last.fm/MuSe → prévias do Deezer) |
| `site/public/` | frontend estático (`index.html`, `app.js`, `styles.css`; `api.js` é o ponto de conexão) |
| `site/api/_vibez/recomendar.py` | o algoritmo (`docs/decisoes.md`, seção 1) (única cópia, usada pela API, pelo notebook e pelos testes) e a versão `clip` do Passo 4, só para a avaliação (`docs/avaliacao.md`, seção 8) |
| `site/api/index.py` | a API (FastAPI na Vercel): CLIP, leitura da foto e contrato (`docs/decisoes.md`, seção 4); dados em `site/api/_dados/` |
| `docs/avaliacao.md`, `avaliacao/` | avaliação humana (Passo 3 do review): protocolo, `gerar.py` (página cega para as avaliadoras), `analisar.py` (α de Krippendorff, Wilcoxon) |
| `tests/` | testes (`python -m unittest discover tests`): algoritmo (`test_recomendar.py`), α de Krippendorff contra o exemplo publicado (`test_alfa.py`) e download do modelo (`test_download.py`) |
| `ferramentas/` | Ferramentas da API (rodar da raiz do repositório): `preparar_textos.py` (gera `textos.npz`), `validar_onnx.py`, `rodar_local.py` (site + API no Mac). A API de verdade fica em `site/api/index.py`. |
