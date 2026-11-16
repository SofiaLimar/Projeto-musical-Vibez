# Vibez repo review

Code, ML and repo-organization review of commit `50c6d69` (16 Nov 2026).

## Verdict

The decision log is unusually good. The README explains each parameter with measured before/after numbers, and the notebooks are honest about their limits. The engineering under it is weaker: nobody can rebuild what is deployed from this repo, and the ranking algorithm is written in two places that already disagree.

The bigger problem is the ML design. Everything is squeezed down to 2 numbers (valence and arousal), and most of the hand-written rules (party mode, dominant genre, the popularity weight) exist to make up for what that loses.

Severity labels: **Critical**, **High**, **Medium**, **Low**.

---

## 1. Machine learning and architecture

### Critical: everything goes through a 2-number bottleneck

The photo becomes a 512-number CLIP vector, Ridge turns it into 2 numbers, and songs are found by distance on that 2D map. A beach, a funeral and a quiet bedroom can land on the same point. That is why the code needs:

- party mode, because OASIS rates party photos as calm;
- a "dominant genre" second pass, because photos near the centre of the map pulled in reggae, k-pop and emo together;
- a popularity weight big enough to drive the ranking.

The project already has the better tool and uses it only in party mode: CLIP text embeddings of the MuSe mood tags (`rótulo_match`). Matching the photo to songs directly in CLIP's space would keep the meaning that 2 numbers throw away.

### Critical: nothing measures whether the playlist matches the photo

Every parameter was tuned on stand-in numbers over OASIS photos: share of popular songs, average distance, genres per playlist. None of them measures relevance, so pushing them up can make playlists worse unnoticed. The README admits it has not been tested with people.

50 photos rated by 3 people on a 1–5 "does it match?" scale, comparing the current version with a CLIP-tags version, would be worth more than all the tuning sweeps. [IMEMNet](https://github.com/linkAmy/IMEMNet) (140K image–music pairs with valence/arousal labels) could serve as an automatic test set.

### High: popularity outweighs mood

The score is `distance − 2 × pop`, with pop between 0.70 and 1.00. The popularity term can move a song by 0.6, while distances among the 90 candidates differ by about 0.34 (median). Measured on `site/api/_dados/musicas.json` (2,812 songs) over 500 random photo points drawn from N(0, 1) on the z-scale:

| The final top 15 shares, on average… | Songs (of 15) |
|---|---|
| with "sort the 90 candidates by popularity only" | 9.5 |
| with "sort the 90 candidates by distance only" | 6.4 |

So the ranking is really "the most popular of the 90 nearest". If that is the intent, write it that way. If not, put both terms on the same scale (for example, convert both to ranks) before weighting them.

### Medium: the common scale (z-scores) is a design choice, not something measured

Each side is rescaled by its own spread, which forces "a typical photo" to mean "a typical song". That is circular. Three smaller points:

- The songs' two axes are correlated (r = 0.47 after filtering; the README says 0.38). Plain Euclidean distance ignores this; a correlation-aware distance (Mahalanobis) would not.
- `percentil()` assumes the predictions follow a bell curve. An empirical percentile from the 900 out-of-sample predictions is more honest and costs one line.
- `mesmo_quadrante` uses `np.sign`, which returns 0 at exactly 0. Harmless, but sloppy.

### Medium: the song scores rest on a weak base model

MusiCNN trained on emoMusic: 744 clips from 2013. The project then removes heavy metal by hand because the model gets it wrong, using the same tags it later evaluates against (the notebook warns about this, which is good).

- Use a newer audio model: Essentia's DEAM heads, or a CLAP or MERT encoder.
- Combine it with the valence/arousal values already in `muse_v3.csv`. The repo ships 18 MB of that data and never uses the per-song scores.

### Medium: the photo model learns from a narrow source

900 OASIS images, much of it extreme content. The README admits this. Training on everyday photos as well (EMOTIC, FI or Emotion6) matters more than tuning Ridge.

### Medium: party mode is a hand-written rule with five magic constants

Threshold 0.5, target (0.5, 1.0), the >0.3 energy filter, weight 0.5 and top 15. The notebook already names the right fix: learn danceability from the audio.

---

## 2. Code-level problems

### Critical: the deployed model cannot be rebuilt from the repo

`foto_parte1_treino.ipynb` never computes `media_prev`/`desvio_prev`. Part 2 checks for them with an assertion that says to "run cell 6 of Part 1 again with the two new lines", and those lines are not committed. The music scoring notebook (`notas_essentia.ipynb`, which produces `notas_musicas.csv`) is missing entirely. The core of the project lives only in someone's Google Drive.

### Critical: the ranking logic is copied in two places, and the copies have drifted

It exists in the notebook (`playlist_final`) and in `site/api/index.py`, including the genre mapping, the artist-name regex, party mode and fallback. The notebook filters party genres by Deezer genre names while the API uses the site's names, and the notebook has no `generos_com_fallback`. The fix is one `vibez/recomendar.py` module that both import.

### High: the README and the code disagree on party mode

The README says `distance − 0.5×pop − 0.5×rótulo_match`. The code (`site/api/index.py:200`) uses `ALFA = 2`.

### High: the model download is fragile

- It downloads the fp32 ONNX file (352 MB) to `/tmp` on every cold start.
- It only checks the file size, not a sha256 hash.
- `urlopen(timeout=50)` limits each read, not the whole download, so it can run past the 60 s `maxDuration`.

Xenova also publishes a quantized `vision_model_quantized.onnx` (~88 MB). Validate it with `validar_onnx.py` and ship that instead.

### High: dependencies are not pinned

`site/requirements.txt` has no versions, so a new numpy or onnxruntime release can silently change production. The data pipeline has no requirements file at all.

### Medium: broken or awkward tooling

- `servidor/validar_onnx.py:21` reads `servidor/dados/modelo.json`, which does not exist. The real file is `site/api/_dados/modelo.json`.
- `celula_exportar_colab.py` is a script meant to be pasted into Colab and depends on its global variables (`PASTA`, `pac`, `musicas`, `Zm`). It should be a function, or a notebook cell.
- The 4 data-collection scripts repeat the same rate-limited Deezer client. `buscar_fas_artistas.py` and `buscar_generos_albuns.py` are the same script with a different endpoint. One `deezer.py` would replace about 150 lines.

### Low: small things

- `assert` is used for input validation (`site/api/index.py:264`); Python removes asserts when run with `-O`.
- `nota_rotulos` runs a Python loop over 2,812 songs on every request, even outside party mode.
- The genre list is written in both `api.js` and `index.py`.
- The center crop on portrait phone photos throws away about 44% of the frame.

### Done well

- Ridge exported as plain weights, checked to 1e-6, so the server doesn't depend on scikit-learn.
- ONNX checked against PyTorch.
- Cross-validation grouped by photo theme, so near-identical photos can't leak between training and test.
- Noise ceiling of the human ratings computed.
- Frontend safe from script injection (`textContent` everywhere).
- Scrapers resume where they stopped.
- Decision log with measured numbers.

---

## 3. Repo organization

- **The root folder is flat** and mixes scrapers, notebooks, a stray `foto_parte1_treino - cópia.ipynb` and 31 MB of CSVs. The "cópia" notebook has saved outputs, including the filename of a photo someone uploaded. Delete it.
- **Data:** `muse_v3.csv` (18 MB) is a third-party dataset redistributed without checking its license. Raw and derived versions are both committed (`popularidade.csv` and `popularidade_final.csv`), and `lista_rotulos_1000.csv` (5.8 MB) is an intermediate file. Use Git LFS or a download script, and keep only final artifacts.
- **The folder names are confusing.** `servidor/` ("server") holds tools and doesn't run as a server; the real server is `site/api/`.
- **Paths point to two different Google Drive folders** (`mapa-sonoro/` and `playlist-foto/`), so the pipeline only runs for whoever owns them.
- **There are no tests.** A small `test_recomendar.py` could check that a fixed embedding gives 15 tracks with no repeated artist, and that party mode triggers. That would catch the drift above.
- **The history is 2 commits**, one of them 248k lines. The README is the only record of how decisions changed.
- **The README mixes a spec, a changelog and research notes.** The content is strong. Split it into `README.md` (what it is, how to run it) and `docs/decisoes.md` (the decision log).

Suggested layout:

```
vibez/          # one package: recomendar.py, clip_onnx.py, deezer.py, config.py
pipeline/       # 01_coletar.py … 05_exportar.py (scripts, no Drive globals)
notebooks/      # exploration only, outputs stripped, imports vibez/
site/           # public/ + api/index.py (thin: imports vibez.recomendar)
data/README.md  # how to fetch; artifacts via LFS or a release
tests/
```

---

## 4. Comparison with similar projects

| Project | Photo becomes | Songs are | Matching | Compared with Vibez |
|---|---|---|---|---|
| **Vibez** | CLIP → Ridge → 2D mood/energy | MusiCNN valence/arousal + popularity + genre | kNN in 2D + hand rules | Strongest decision log and validation of the group |
| [Image→Song Recommender](https://huggingface.co/Jeevaaaa/image-to-song-recommender) | CLIP vs 12 mood prompts (no training) | 5 Spotify audio features | kNN | Simpler, and uses more than 2 dimensions |
| [Snapify](https://github.com/vickydev810/snapify) | BLIP caption → text emotion classifier | Spotify API | API recommendations | Weaker: emotion read from a caption, not the image |
| [Image2Playlist](https://github.com/YuktiVasuja/Image2Playlist) | VGG16 features | Spotify | — | Weaker: older approach |
| [Emo-CLIM](https://arxiv.org/html/2308.12610v3) | Frozen CLIP + learned projection | CLAP/CNN audio → shared space | Contrastive retrieval | Ahead: a learned image–music space with retrieval metrics (P@5 ≈ 68%) |
| [IMEMNet / CDCML](https://arxiv.org/abs/2009.05103) | Image valence/arousal | DEAM music valence/arousal | Learned image–music matching | Ahead: 140K labelled pairs to evaluate against |

Vibez is clearly ahead of the hobby projects on rigour: grouped cross-validation, noise ceiling, before/after numbers. It is behind the research on architecture. The field has moved from "predict 2 numbers, then nearest neighbour" to learning a shared image–music space.

---

## 5. What to do, in order

1. **Make it reproducible.** Commit the missing training lines and the music scoring notebook, pin dependencies, and fix the `validar_onnx` path.
2. **Keep one copy of the algorithm** (`vibez/recomendar.py`), and add one test.
3. **Build a small human evaluation set** before tuning any more parameters.
4. **Try a CLIP mood-tag retrieval baseline:** `score = cosine(photo, song tags) + λ·valence/arousal distance + μ·rank(pop)`. Compare it with the current version on that evaluation set. If it wins, party mode and the dominant-genre pass can probably be deleted.
5. **Longer term:** CLAP audio embeddings for the songs plus a small trained projection, in the style of Emo-CLIM.

---

## Sources

- [Image→Song Recommender (Hugging Face)](https://huggingface.co/Jeevaaaa/image-to-song-recommender)
- [Snapify](https://github.com/vickydev810/snapify)
- [Image2Playlist](https://github.com/YuktiVasuja/Image2Playlist)
- [Emo-CLIM: Emotion-Aligned Contrastive Learning Between Images and Music](https://arxiv.org/html/2308.12610v3)
- [Zhao et al., Emotion-Based End-to-End Matching Between Image and Music](https://arxiv.org/abs/2009.05103) · [IMEMNet data](https://github.com/linkAmy/IMEMNet)
- [LAION CLAP](https://github.com/LAION-AI/CLAP)
- [Sasaki et al., SIGGRAPH 2013 poster](https://history.siggraph.org/wp-content/uploads/2022/12/2013-Poster-15-Sasaki_Affective-Music-Recommendation-System-using-Input-Images.pdf)
