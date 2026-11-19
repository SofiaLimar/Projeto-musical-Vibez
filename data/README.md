# Dados

De onde vem cada arquivo desta pasta, quem usa e o que é final ou intermediário.
Os comandos dos scripts rodam na raiz do repositório (ordem completa no `README.md`, "Refazer os dados").

## Arquivos em `data/`

| Arquivo | Tamanho | Gerado por | Usado por | Tipo |
|---|---|---|---|---|
| `muse_v3.csv` | 18 MB | terceiros: MuSe (ver licença abaixo) | `pipeline/contar_rotulos.py` | entrada externa |
| `lista_rotulos_1000.csv` | 5,7 MB | `pipeline/contar_rotulos.py` (Last.fm) | `pipeline/baixar_audio.py` | intermediário |
| `audio_teto100.csv` | 2,5 MB | `pipeline/baixar_audio.py` (Deezer) | `pipeline/buscar_popularidade.py`, notebooks | **final** (índice das prévias) |
| `notas_musicas.csv` | 0,9 MB | `notebooks/notas_essentia.ipynb` (Colab) | `notebooks/foto_parte2_playlist.ipynb` | **final** |
| `dancabilidade.csv` | 0,1 MB | `notebooks/notas_essentia.ipynb`, Célula 8 (Colab) | `notebooks/foto_parte2_playlist.ipynb` (seção 5b) | **final** |
| `popularidade.csv` | 0,4 MB | `pipeline/buscar_popularidade.py` (Deezer) | `buscar_fas_artistas.py`, `buscar_generos_albuns.py`, `montar_popularidade.py` | intermediário |
| `fas_artistas.csv` | 0,1 MB | `pipeline/buscar_fas_artistas.py` (Deezer) | `pipeline/montar_popularidade.py` | intermediário |
| `generos_albuns.csv` | 0,2 MB | `pipeline/buscar_generos_albuns.py` (Deezer) | `pipeline/montar_popularidade.py` | intermediário |
| `popularidade_final.csv` | 0,8 MB | `pipeline/montar_popularidade.py` | `notebooks/foto_parte2_playlist.ipynb` | **final** |
| `cache_lastfm/` | 9,8 MB | `pipeline/contar_rotulos.py` | o próprio script (retomada) | cache, fora do git |

Os intermediários servem para **retomar** uma coleta interrompida: os scripts leem o que já foi feito e continuam.
O site em si usa só `site/api/_dados/` (`modelo.json`, `musicas.json`, `textos.npz`).

## Licença do MuSe (`muse_v3.csv`)

MuSe: The Musical Sentiment Dataset (Akiki e Burghardt, Universidade de Leipzig).

- **Origem:** Kaggle, `https://www.kaggle.com/datasets/cakiki/muse-the-musical-sentiment-dataset` (conta "cakiki", de Christopher Akiki,
  primeiro autor), arquivo `muse_v3.csv`, baixado em 13/11/2026. Tem 90.001 músicas, o número do artigo de 2021.
- **Licença:** Creative Commons Attribution 4.0 (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/, conferida na página do Kaggle em 18/11/2026.
  Permite redistribuir (inclusive neste repositório) desde que os autores sejam citados.
- **Citação:** Akiki, C. e Burghardt, M. (2021). MuSe: The Musical Sentiment Dataset. *Journal of Open Humanities Data*, 7. https://doi.org/10.5334/johd.33
- A versão do Zenodo (https://zenodo.org/records/4281165, também CC BY 4.0) é outro arquivo (`muse_dataset.csv`, 90.408 músicas):
  não troque um pelo outro sem refazer o pipeline.

As tabelas coletadas do Deezer e do Last.fm vêm das APIs públicas desses serviços.

## Pastas no Google Drive (notebooks)

Os notebooks rodam no Colab e usam duas pastas do Drive. Os caminhos ficam só na primeira célula de cada notebook;
quem for rodar em outra conta precisa criar estas pastas (ou mudar os caminhos nessa célula):

| Pasta no Drive | O que precisa ter | Quem usa |
|---|---|---|
| `mapa-sonoro/` | `oasis/` (baixado pela parte 1), `modelos/` (embeddings do OASIS e `foto_humor_energia.joblib`), `popularidade_final.csv`, `audio_teto100.csv` | `foto_parte1_treino`, `foto_parte2_playlist` |
| `playlist-foto/` | `audio_teto100.csv`; recebe `notas_musicas.csv`, `embeddings/` e `dancabilidade.csv` | `notas_essentia`, `foto_parte2_playlist` |

As prévias de áudio (`audio_previas/`, ~12 mil MP3) não ficam no git nem no Drive: o `notas_essentia.ipynb` baixa cada prévia do Deezer na hora.
