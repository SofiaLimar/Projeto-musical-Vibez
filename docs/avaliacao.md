# Avaliação humana: a playlist combina com a foto?

Passo 3 do review (`docs/REVIEW.md`): antes de ajustar qualquer parâmetro, medir **relevância** com pessoas.
Até aqui os parâmetros foram ajustados com medidas indiretas (popularidade, distância, gêneros por playlist),
e nenhuma delas diz se a playlist combina com a foto.

Este documento fixa as regras **antes** de ver qualquer resultado. Se algo mudar depois, registre aqui o quê e por quê.

## 1. A pergunta

> **"Quanto esta playlist combina com o clima da foto?"**

| Nota | Significado |
|---|---|
| 1 | Não combina nada |
| 2 | Combina pouco |
| 3 | Combina mais ou menos |
| 4 | Combina bem |
| 5 | Combina totalmente |

Avalia-se o **clima** (o que a foto passa), não se a pessoa gosta das músicas.
Essa instrução aparece no topo da página de avaliação.

## 2. As fotos

- **50 fotos do dia a dia**, não do OASIS (o modelo foi treinado no OASIS; o review aponta que ele cobre mal fotos comuns).
- Cobrir situações variadas, por exemplo: praia, festa, rua/cidade, natureza, comida, quarto/casa, viagem, show,
  dia chuvoso, noite, esporte, trabalho/estudo, animais. Algumas fotos "difíceis" (ambíguas, sem pessoas, só objetos) também.
- Origem: fotos próprias ou de bancos com licença livre (Unsplash, Pexels). **Nada de rostos de terceiros sem autorização.**
- As fotos ficam fora do GitHub (pasta `avaliacao/fotos/`, ignorada pelo git).
- **Conjunto reservado:** estas 50 fotos servem para comparar versões, não para ajustar parâmetros.
  Se um parâmetro for escolhido olhando o resultado nelas, a medida deixa de valer para esse parâmetro;
  nesse caso, montar um segundo conjunto para a confirmação.

## 3. As pessoas

- **3 avaliadoras(es)**. A autora pode ser uma delas; as outras duas, de preferência, não conhecem o sistema.
- **Cego:** a página mostra "Playlist 1" e "Playlist 2", sem nome de versão, sem o nome da playlist, sem modo festa e sem rótulos.
  Os códigos de cada playlist são sorteados; a ligação código → versão fica só em `chave.json`, que não vai para as avaliadoras.
- **Ordem sorteada:** a ordem das fotos e a posição de cada versão (1 ou 2) são sorteadas para cada pessoa.
- Cada pessoa avalia sozinha, sem conversar com as outras sobre as notas.

## 4. A tarefa

Para cada foto, a pessoa vê a foto e as playlists (15 faixas cada) e:
1. ouve **pelo menos 3 prévias** de cada playlist (a página conta quantas foram ouvidas);
2. dá uma nota de 1 a 5 para cada playlist.

Tempo estimado: 2 a 3 minutos por foto, ou seja, cerca de 2 horas para 50 fotos.
Dá para parar e voltar: a página guarda as notas no próprio navegador.
No fim, a pessoa clica em **Exportar respostas** e envia o arquivo `.csv`.

## 5. A análise (`avaliacao/analisar.py`)

1. **Concordância entre as pessoas:** α de Krippendorff, ordinal (Krippendorff, 2004).
   Referência usual: α ≥ 0,80 confiável; 0,667 ≤ α < 0,80 só para conclusões provisórias; α < 0,667 não conclusivo.
   Se o α ficar baixo, a pergunta ou as instruções precisam ser revistas antes de comparar versões.
2. **Nota de cada versão:** média das notas.
3. **Comparação entre duas versões** (Passo 4): para cada foto, a média das 3 notas de cada versão;
   diferença B − A por foto; **teste de Wilcoxon pareado** nas 50 diferenças e **intervalo de 95% por bootstrap** da diferença média.
4. Conferência de esforço: quantas prévias cada pessoa ouviu por playlist.

## 6. Regra de decisão (definida antes, a confirmar)

A versão nova substitui a atual se:
- o α ordinal for ≥ 0,667; **e**
- a diferença média (nova − atual) for positiva, com o intervalo de 95% inteiro acima de zero.

Se o intervalo cruzar o zero, as duas versões são consideradas equivalentes nesta avaliação.

## 7. Ordem de execução

- **Passo 3 (agora):** escolher as 50 fotos e fazer um **piloto** com 5 fotos e só a versão atual,
  para testar a página, o tempo da tarefa e se a pergunta está clara.
- **Passo 4:** gerar as playlists da versão atual e da versão com tags do CLIP para as 50 fotos,
  aplicar com as 3 pessoas e analisar. As duas versões ficam na mesma API (campo `versao` = `atual` ou `clip`);
  o site continua usando só a atual.

## 8. A versão comparada no Passo 4 (`versao="clip"`)

Fórmula do review (seção 5, item 4): `cosseno(foto, rótulos da música) + λ · distância valência/energia + μ · rank(pop)`.
O review não fixa os detalhes; estas escolhas foram feitas **antes** da avaliação e não podem ser ajustadas olhando as 50 fotos:

- **Mesma escala:** os três termos viram posição percentil entre as músicas (0 a 1), como o review sugere para a popularidade.
  A distância entra invertida (mais perto = posição maior).
- **Pesos:** λ = μ = 1.
- **Cosseno com os rótulos:** a média dos cossenos entre a foto e os rótulos do MuSe da música
  (frase `a photo with a {rótulo} mood`, os mesmos embeddings de `textos.npz`).
  Para 88% das músicas (1 rótulo só) isso é o mesmo que o maior cosseno. A média foi escolhida (18/11, antes da avaliação)
  porque o maior cosseno favorece as músicas com mais rótulos: medido com as 13 frases de cena de `textos.npz`
  como fotos de referência (não as fotos da avaliação), a posição média das músicas com 2–5 rótulos subia de 0,48
  para 0,61–0,74 com o maior cosseno; com a média, ficava entre 0,31 e 0,46.
- **Sem modo festa e sem gênero dominante:** o review diz que, se esta versão ganhar, essas regras provavelmente podem sair.
- Continuam: no máximo 1 música por artista, 15 faixas, e o filtro de gênero quando a pessoa escolhe um gênero.

## Referência

Krippendorff, K. (2004). *Content Analysis: An Introduction to Its Methodology* (2ª ed.). Sage. Cap. 11 (confiabilidade).

## 9. Resultado (18/11/2026)

Estudo `481ada02`: 52 fotos, 2 versões (`atual` e `clip`), 3 pessoas (a autora e mais duas), 312 notas.
Saída de `avaliacao/analisar.py`:

| Medida | Valor |
|---|---|
| α de Krippendorff ordinal (intervalar) | **0,101** (0,165): não conclusivo |
| Nota média, versão atual | 4,11 (desvio 1,06; 156 notas) |
| Nota média, versão clip | 3,51 (desvio 1,43; 156 notas) |
| Diferença clip − atual (média por foto) | **−0,60**, IC 95% (bootstrap) [−0,91; −0,30], Wilcoxon p = 0,001 |
| Fotos em que clip foi melhor / pior / empate | 13 / 30 / 9 |

**Decisão pela regra da seção 6:** a versão `clip` **não substitui** a atual: o α ficou abaixo de 0,667 e o intervalo
ficou abaixo de zero (não acima). O site, o modo festa e a etapa de gênero dominante continuam como estão.

**Como ler:** com α = 0,10 as notas de cada playlist variaram muito entre as pessoas, então o resultado não permite
afirmar com segurança que a atual é melhor; permite dizer que a versão `clip`, como definida na seção 8, não mostrou vantagem.

**Leitura exploratória (fora da regra, só para orientar o próximo passo):**
- as três pessoas, cada uma, deram em média notas menores à `clip` (clip − atual: −0,37, −0,90 e −0,52);
- foto a foto, os pares de pessoas concordaram sobre qual versão foi melhor em 14, 25 e 26 das 52 fotos;
- uma pessoa deu nota 5 em 70% das playlists (efeito teto), o que puxa o α para baixo; cada pessoa usou a escala de um jeito.

**Desvios do protocolo:**
- a autora conhece o sistema (seção 3 pedia, de preferência, pessoas que não conhecem);
- 21 das 104 playlists da autora e 2 das 104 de outra pessoa tiveram menos de 3 prévias ouvidas (seção 4);
- dois arquivos de resposta foram abertos e salvos no Numbers: o separador virou ponto e vírgula e o código `6635e4`
  (foto f24) virou `6,635E+07`. Recuperado sem ambiguidade (era o único código da f24 ausente no arquivo e o valor numérico bate);
  um terceiro arquivo foi salvo com cada linha entre aspas, desfeito sem perda. Os originais ficam em `avaliacao/respostas/originais/`.

**Consequências para os próximos passos:**
- estas 52 fotos **não** podem ser usadas para ajustar a versão `clip` (pesos λ e μ, rótulos etc.); um teste de variação
  pede um segundo conjunto de fotos (seção 2);
- antes da próxima comparação, rever a pergunta e as instruções (seção 5, α baixo): por exemplo, explicar melhor a escala
  para reduzir o efeito teto, e reforçar o mínimo de prévias.
