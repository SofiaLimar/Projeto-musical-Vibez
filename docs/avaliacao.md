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
- **Passo 4:** gerar as playlists da versão atual e da versão com tags do CLIP (publicada num deploy de prévia)
  para as 50 fotos, aplicar com as 3 pessoas e analisar.

## Referência

Krippendorff, K. (2004). *Content Analysis: An Introduction to Its Methodology* (2ª ed.). Sage. Cap. 11 (confiabilidade).
