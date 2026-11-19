# vibez: registro de decisões

Como o algoritmo, os dados e o site chegaram ao que são, com os números medidos antes e depois de cada mudança.
Movido do `README.md` em 18/11/2026 (sugestão do review, `docs/REVIEW.md` seção 3); o conteúdo é o mesmo.
As seções mantêm a numeração antiga, que outros arquivos citam ("README, seção 1" = seção 1 deste documento).

---

## 1. Como a playlist é montada (algoritmo exato)

```
foto
 └─ CLIP ViT-B/32 (openai/clip-vit-base-patch32): vetor de 512, normalizado (norma 1)
     └─ StandardScaler + Ridge (2 saídas), treinado no OASIS: (valência, energia) na escala OASIS 1–7
         └─ z da foto = (previsão − media_prev) / desvio_prev
             └─ busca no plano z das músicas (já filtradas: limpeza + popularidade + gênero)
                 ├─ pega as 90 mais próximas (15 × 6) pela distância euclidiana
                 ├─ nota = distância − 2 × pop          (menor = melhor)
                 ├─ no máximo 1 música por artista-base e nenhum deezer_id repetido
                 ├─ sem gênero escolhido: acha o gênero dominante dessa playlist provisória
                 │   e refaz com esse gênero + 1 vizinho, olhando as 30 mais próximas
                 └─ as 15 melhores notas → playlist
(foto de festa: regra própria, ver "Modo festa")
```

| Parâmetro | Valor | Por quê |
|---|---|---|
| `POP_MIN` | 0,7 | Abaixo disso aparecem artistas que ninguém conhece (checado à mão nos percentis 30/50/70%). |
| `TAMANHO_PLAYLIST` | 15 | |
| `CANDIDATOS_POR_VAGA` | 6 | 90 candidatas |
| `ALFA` | **2** (era 0,5) | Mudado em 15/11 após teste com usuária: com 0,5 a popularidade quase não pesava (pop só vai de 0,7 a 1). Varredura em 900 fotos OASIS, com o limite de 90 candidatos: músicas no top 10% de pop 39% → 73%, fãs medianos 837 mil → 1,9 mi, distância média 0,19 → 0,25 (p90 0,41 → 0,47), catálogo usado 79% → 44%, música mais repetida 52 → 90 de 900. Subir POP_MIN para 0,85/0,9 foi pior (abre buracos no mapa, p90 0,56–0,67). |
| `CANDIDATOS_GENERO` | 30 | **Novo (15/11).** Modo normal *sem gênero escolhido*: monta uma playlist provisória, pega o gênero do site mais comum nela e refaz só com esse gênero + o 1º vizinho (tabela de fallback), olhando as 30 mais próximas. Motivo: foto no centro do mapa (ex.: humor 46, energia 37) tem todas as 90 candidatas à mesma distância, e a popularidade sozinha misturava reggae, k-pop e emo. Varredura OASIS: gêneros por playlist 5,9 → 1,9; distância 0,25 → 0,29 (p90 0,47 → 0,57); pop 0,92 → 0,89. Se o usuário escolhe gênero, vale a escolha dele. |
| Repetição | — | **Corrigido (15/11):** além de 1 música por artista, ignora `deezer_id` repetido e usa o artista-base ("Bob Marley & The Wailers" = "Bob Marley"; corta em ` & `, ` feat`, ` ft`, `, `, ` and `). |
| `MAX_POR_ARTISTA` | 1 | |

**Na prática, a popularidade decide mais que a distância (intencional).** Com `ALFA = 2`, o termo de popularidade (pop entre 0,7 e 1) pode mover a nota de uma música em até 0,6, enquanto as distâncias entre as 90 candidatas diferem em cerca de 0,34 (mediana). O ranking é, portanto, **"as mais populares entre as 90 mais próximas"**: a distância escolhe quais músicas concorrem, e a popularidade decide a ordem entre elas. Medido no review (`docs/REVIEW.md`, seção 1, "popularity outweighs mood"), em `site/api/_dados/musicas.json` (2.812 músicas) com 500 pontos de foto sorteados de N(0, 1) na escala z: o top 15 final tem em média **9,5** músicas em comum com "ordenar as 90 só por popularidade" e **6,4** com "ordenar só por distância". É o efeito buscado ao subir `ALFA` de 0,5 para 2 (ver a linha `ALFA` acima): playlists com músicas conhecidas, dentro do clima da foto. A alternativa do review (colocar distância e popularidade na mesma escala antes de dar os pesos) não foi adotada: a intenção é priorizar músicas conhecidas dentro do clima da foto, e os pesos ficam como estão.

### Modo festa (regra provisória)

O modelo de foto erra fotos de festa: no OASIS, *Party*, *Drink* e *Alcohol* têm arousal de ~2 a 3, ou seja, são consideradas "calmas". Já os rótulos que o CLIP associa a essas fotos são claros (*celebratory, euphoric, raucous*). A regra é:

1. **Classificar a cena com o CLIP**, comparando o embedding da foto com 13 frases e aplicando softmax com escala 100:
   - festa (4): `a photo of a party`, `a photo of people dancing at a nightclub`, `a photo of a celebration with friends`, `a photo of a concert crowd`
   - outras (9): `a photo of a landscape`, `a photo of a beach`, `a portrait of a person`, `a photo of a city street`, `a photo of nature`, `a photo of food`, `a photo of a room`, `a selfie`, `a photo of animals`
   - `prob_festa` = soma das probabilidades das 4 frases de festa.
2. **Se `prob_festa ≥ 0,5`:**
   - o alvo vira `(max(zv, 0,5), max(ze, 1,0))`, ou seja, é puxado para alegre + agitada;
   - os gêneros são os escolhidos pelo usuário ou, se ele não escolheu nenhum, Pop, Dance, Electro, Rap/Hip Hop e R&B;
   - só entram músicas com `zv > 0` e `ze > 0,3`. Se sobrarem menos de 30, esse filtro de humor e energia é retirado;
   - a nota vira `distância − ALFA × pop − 0,5 × rótulo_match` (`ALFA` = 2, o mesmo do modo normal), com no máximo 1 música por artista.
3. **Senão:** o algoritmo normal, **sem** o termo de rótulos.

**`rótulo_match`** existe para que duas festas diferentes não recebam a mesma playlist. Sem ele, os alvos coincidiam e as playlists repetiam quase 15 de 15 músicas; com ele, repetiram 4 de 15. O cálculo:
- compara a foto com cada rótulo MuSe da base usando a frase `a photo with a {rótulo} mood` (similaridade de cosseno no CLIP);
- pega os 15 rótulos mais parecidos e dá a cada um o peso `exp(100 × (sim − sim_max))`, de modo que o melhor rótulo vale 1;
- dá a cada música o maior peso entre os rótulos dela (coluna `rotulos`, separada por "|"). Se nenhum rótulo dela estiver entre os 15, vale 0.

Exemplo: a festa num bar recebeu *hedonistic, hypnotic, trippy, flashy*; a de ano-novo, *celebratory, euphoric, ecstatic, fun*.

Os embeddings das 13 frases de cena e dos rótulos são pré-calculados (`site/api/_dados/textos.npz`, gerado por `ferramentas/preparar_textos.py`). Assim a API só precisa do encoder de imagem.

**Limitação:** é uma regra manual, não aprendida.

**Régua (z-score):** cada lado é padronizado pela **própria** distribuição. Uma foto "entre as mais calmas" recebe "músicas entre as mais calmas".
- **Foto:** usa `media_prev` e `desvio_prev`, que são a média e o desvio das **previsões** fora da amostra, e não das notas humanas. As previsões são ~22% mais comprimidas que as notas humanas; com a régua humana, as fotos nunca chegariam aos extremos.
- **Música:** usa a média e o desvio das músicas **depois** dos filtros de limpeza e popularidade. O filtro de gênero **não** muda a régua.

---

## 2. Gênero (decidido)

- **Opcional.** O padrão é **"Qualquer"**. Pode-se marcar mais de um.
- **Só 7 opções.** Os rótulos de cima são o que aparece no site; os de baixo são os gêneros do Deezer, que é o valor da coluna `genero`.

| Opção no site | Gêneros Deezer |
|---|---|
| Pop | `Pop` |
| Rock | `Rock` |
| Alternativo | `Alternativo` |
| R&B | `R&B` |
| Rap/Hip Hop | `Rap/Hip Hop` |
| Eletrônica | `Electro`, `Dance` |
| Trilhas sonoras | `Filmes/Games` |

- **Fora:** Jazz e Metal (51 músicas ou menos, buracos em 3 de 4 climas) e Música asiática (buracos). A música brasileira praticamente não existe na base.
- **Fallback:** se os gêneros escolhidos tiverem **menos de 15 músicas no quadrante da foto**, completar com o gênero vizinho e **avisar** ("completamos com Pop e Alternativo"). Os quadrantes são definidos pelo sinal de z: alegre+agitada, alegre+calma, triste+agitada, triste+calma.

| Escolhido | Completar com |
|---|---|
| Pop | R&B, Alternativo |
| Rock | Alternativo, Pop |
| Alternativo | Rock, Pop |
| R&B | Pop, Rap/Hip Hop |
| Rap/Hip Hop | R&B, Pop |
| Eletrônica | Pop, Alternativo |
| Trilhas sonoras | Alternativo, Pop |

Músicas por gênero × quadrante (base com pop ≥ 0,7). Os buracos (< 15) estão em **negrito**:

| | alegre+agitada | alegre+calma | triste+agitada | triste+calma |
|---|---|---|---|---|
| Pop | 355 | 128 | 58 | 260 |
| Rock | 175 | 55 | 73 | 133 |
| Alternativo | 175 | 83 | 64 | 259 |
| R&B | 55 | 41 | **11** | 62 |
| Rap/Hip Hop | 32 | **11** | 124 | 33 |
| Electro (sem Dance) | 51 | **11** | **8** | 35 |
| Filmes/Games | 31 | 17 | **9** | 91 |

---

## 3. Dados

| Arquivo | Onde | O que tem |
|---|---|---|
| `notas_musicas.csv` | Drive `playlist-foto/` | `deezer_id`, `valencia`, `energia` (MusiCNN + emoMusic, escala ~1–9) e `status` (`ok` ou `sem_preview`) |
| `data/audio_teto100.csv` | Mac e Drive | `deezer_id`, `artista`, `titulo`, `rotulos` (MuSe, separados por "\|"), `data_lancamento`, `arquivo` |
| `data/popularidade_final.csv` | Mac e Drive | `deezer_id`, `rank`, `nb_fan`, `pop` (0–1), `genero`, `generos`, `genero_origem` |
| `modelos/foto_humor_energia.joblib` | Drive `mapa-sonoro/` | o modelo de foto, com `media_prev`, `desvio_prev`, `media_oasis` e `desvio_oasis` |

**Limpeza obrigatória**, feita antes da popularidade, na célula 3c do notebook. O MusiCNN erra com metal extremo (valência fora da escala, energia baixa). São removidas:
1. as músicas com `valencia < 1`;
2. as músicas com rótulo agressivo (`aggressive, angry, hostile, menacing, harsh, brash, volatile, fiery`) e `energia` abaixo da mediana.

Depois da limpeza e do `POP_MIN`, sobram **2.812 músicas**.

**Para o site:** exportar um `musicas.json` só com as músicas que passaram nos filtros. Cada item leva `deezer_id`, `artista`, `titulo`, `genero` (já convertido para as 7 opções), `zv`, `ze` (já padronizados) e `pop`. São menos de 1 MB.

---

## 4. Contrato com o frontend (`site/api.js`)

O `api.js` atual espera `POST {API_URL}/playlist` (multipart, campo `foto`). Ajustes decididos:

- **Entrada:** acrescentar o campo `generos` (lista; vazia = "Qualquer").
- **Gêneros:** o site pede a lista em `GET {API_URL}/generos` ao abrir; a lista fica só em `site/api/_vibez/recomendar.py` (antes estava repetida no `api.js`).
- **Saída:** acrescentar `aviso` (texto do fallback ou `null`) e, em cada faixa, `genero`.
- **Escala:** o `api.js` antigo assumia `valencia`/`energia` de **1 a 9**, mas o modelo de foto prevê na escala **OASIS 1–7**. Para a interface, mandar **percentil** (0–100) ou a versão em z. Não converter 1–7 para 1–9 direto, porque as previsões são comprimidas e quase nada chegaria às pontas.
- **Prévias:** **não** dá para publicar as 12 mil prévias na Vercel (~6 GB). Duas alternativas:
  - pegar o `preview` na hora em `https://api.deezer.com/track/{deezer_id}` (a API não libera CORS; usar JSONP com `&output=jsonp`);
  - ou usar o player embutido do Deezer: `https://widget.deezer.com/widget/dark/track/{deezer_id}`.

  Os links de prévia do Deezer expiram, então não dá para guardá-los.

### Onde roda o modelo: **servidor (decidido em 14/11/2026)**

Escolhida a opção 2, mas **tudo na Vercel** (o Docker do Hugging Face Space exige conta PRO): o site estático e a API Python (`site/api/index.py`, FastAPI) ficam no mesmo projeto Vercel "vibez" (https://vibez-azure.vercel.app; publicar com `npx vercel --prod` dentro de `site/`). O CLIP roda em ONNX (validado: r = 1,0000 contra o PyTorch). A tabela abaixo fica como registro da comparação.
- O Ridge vai para o servidor como pesos (`modelo.json`, gerado pela seção 5b de `notebooks/foto_parte2_playlist.ipynb`), sem depender da versão do scikit-learn.
- A resposta manda `valencia_pct`/`energia_pct` (percentil Φ(z) × 100), `aviso`, `genero` por faixa, `modo_festa` e `rotulos`.
- As prévias são pedidas pelo site ao Deezer (JSONP) só quando a pessoa aperta play.

| | Opção 1 (recomendada): navegador | Opção 2: servidor |
|---|---|---|
| Como | CLIP em ONNX via Transformers.js (`Xenova/clip-vit-base-patch32`); Ridge exportado como 512×2 pesos + intercepto; busca em JS | FastAPI ou Gradio num Hugging Face Space, com o mesmo código Python do Colab |
| Hospedagem | Vercel, site estático e grátis | Vercel (frontend) + Space (API) |
| Prós | Sem servidor; a foto não sai do aparelho | Idêntico ao Colab |
| Contras | ~90 MB baixados no 1º acesso; **precisa testar** se as previsões batem com o Python (meta: r > 0,98 nas 900 fotos do OASIS) | O Space grátis "dorme": 30–60 s na primeira foto |

A Vercel não comporta o PyTorch (limite de 250 MB), por isso a API usa **ONNX Runtime**. O encoder de imagem (`vision_model.onnx`, 352 MB, repositório público Xenova, versão fixada) é baixado para `/tmp` quando a função acorda, com conferência de tamanho e **sha256** e prazo de 45 s para o download inteiro (a função tem 60 s). O arquivo quantizado do mesmo repositório (`vision_model_quantized.onnx`, 89 MB), sugerido pelo review, foi **reprovado** em 18/11/2026 no `ferramentas/validar_onnx.py` (`notebooks/validar_onnx_quantizado.ipynb`, 200 fotos do OASIS, contra o CLIP em PyTorch):

| Modelo | r humor | r energia | cosseno mín. | dif. máx. de previsão |
|---|---|---|---|---|
| `vision_model.onnx` (fp32, 352 MB) | 1,0000 | 1,0000 | 0,9999 | 0,012 |
| `vision_model_quantized.onnx` (89 MB) | 0,9738 | 0,9829 | 0,7723 | 1,184 |

A meta era r > 0,98 nos dois; o humor ficou abaixo, e a diferença de até 1,18 na previsão muda playlists. O site continua com o fp32 (`MODELO_CLIP = "fp32"`). A primeira foto depois de um tempo parado pode levar ~20 s; o site chama `/api/saude` ao abrir para adiantar isso.

O Ridge vira uma conta só: `w = coef / scale`, `b = intercept − Σ(coef × mean / scale)`, aplicada ao embedding **normalizado** (`site/api/_dados/modelo.json`).

### Site: decisões de 15/11/2026

- **Sem botão "Ouvir no Deezer" no topo da playlist.** O site não cria playlist no Deezer, então o botão só abria a 1ª música. Fica só o link de cada faixa.
- **Envio pelo celular (iPhone não enviava nada):**
  - um toque abre o seletor de fotos **uma vez** (antes, o botão dentro da área de envio abria duas vezes e o iPhone perdia a foto);
  - o `<input type="file">` fica invisível por CSS (`.input-escondido`), não com `hidden`;
  - aceita qualquer imagem, inclusive **HEIC**; a foto é reduzida para 1024 px **antes** de checar o tamanho (fotos de celular têm 5–15 MB e viram < 1 MB);
  - qualquer falha vira mensagem na tela, e a página rola até ela.
- **Texto da área de envio:** no computador, "Arraste sua foto até aqui / ou anexe uma imagem"; em tela de toque (celular, tablet), só "Anexe uma imagem". A troca usa `@media (hover: none) and (pointer: coarse)`, não a largura.
- **Safari 17 (macOS 14):** os gêneros ficavam por baixo dos cards de "Como funciona". Correção em `#tela-inicio .hero`: `min-height: 0; grid-template-rows: minmax(0, 1fr)`.
- **Testado** (Playwright, 15/11) em WebKit 17, WebKit 18 e Chromium: iPhone SE/13/14 Pro Max, Pixel 7, iPad e telas de 1000×565 a 1920×1080. Nenhuma sobreposição, rolagem lateral ou erro de JS, e a playlist abre com 15 faixas. São navegadores simulados: o seletor de fotos real do iPhone precisa ser testado no aparelho.

---

## 5. Qualidade medida

| O quê | Resultado |
|---|---|
| Foto → humor/energia (OASIS, 5 folds agrupados por tema) | r = 0,83 / 0,82 (paisagens 0,88 / 0,74; retratos 0,83 / 0,72) |
| Comparação de modelos de foto | ResNet18 + Ridge 0,56 / 0,57; CLIP zero-shot 0,77 / 0,19; **CLIP + Ridge 0,85 / 0,82** (escolhido) |
| Notas das músicas contra os rótulos MuSe | AUC energia 0,87, valência 0,81 |
| Cobertura (900 fotos) | distância até a 15ª música: mediana 0,10; 95% ≤ 0,48 |

## 6. Limitações

- Medido com pessoas em 18/11/2026 (52 fotos, 3 pessoas): versão atual 4,11 de 5, mas com concordância baixa entre as pessoas (α = 0,10). Detalhes em `docs/avaliacao.md`, seção 9.
- Nas músicas, humor e energia andam juntos (r = 0,47 nas 2.812 músicas do site, conferido em `site/api/_dados/musicas.json` em 18/11/2026; antes este README dizia 0,38, valor que não bate com os dados atuais); nas fotos, não (−0,06). Fotos "calmas e felizes" e "tensas e agitadas" têm menos músicas disponíveis.
- O modelo de foto erra quando o negativo está no **sentido**, e não na aparência (incêndio bonito, cartaz de ódio).
- O OASIS tem muito conteúdo extremo; com fotos do dia a dia, a precisão provavelmente cai.
- Quase não há música brasileira (a base veio do Last.fm em inglês).
- 1.149 músicas ficaram sem nota (`sem_preview`) e 427 sem gênero.
