(() => {
  const $ = (s) => document.querySelector(s);

  const entrada = $("#arquivo");
  const dropzone = $("#dropzone");
  const erro = $("#erro");
  const player = $("#player");
  let carregando = false;
  let urlFoto = null;
  let linhaAtual = null;
  const escolhidos = new Set();   // gêneros marcados; vazio = "Qualquer"
  const previas = new Map();      // deezer_id -> url da prévia (pedida na hora, expira)

  acordarServidor();

  // ---------- gêneros ----------
  const caixaGeneros = $("#generos");
  let generos = [];               // vem da API (carregarGeneros, em api.js)
  function renderGeneros() {
    caixaGeneros.innerHTML = "";
    ["Qualquer", ...generos].forEach((g) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "gen";
      b.textContent = g;
      const ativo = g === "Qualquer" ? escolhidos.size === 0 : escolhidos.has(g);
      b.setAttribute("aria-pressed", ativo);
      b.addEventListener("click", () => {
        if (g === "Qualquer") escolhidos.clear();
        else if (escolhidos.has(g)) escolhidos.delete(g);
        else escolhidos.add(g);
        renderGeneros();
      });
      caixaGeneros.appendChild(b);
    });
  }
  renderGeneros();
  carregarGeneros().then((lista) => { generos = lista; renderGeneros(); });

  // barras das ondas (decorativas)
  document.querySelectorAll(".ondas").forEach((el) => {
    const n = el.classList.contains("pequenas") ? 7 : 34;
    for (let i = 0; i < n; i++) {
      const b = document.createElement("i");
      b.style.height = `${25 + Math.round(Math.abs(Math.sin(i * 1.7)) * 75)}%`;
      b.style.animationDelay = `${(i % 9) * -0.13}s`;
      el.appendChild(b);
    }
  });

  // ---------- navegação ----------
  function mostrar(tela) {
    $("#tela-inicio").hidden = tela !== "inicio";
    $("#tela-playlist").hidden = tela !== "playlist";
    window.scrollTo({ top: 0 });
  }

  document.addEventListener("click", (e) => {
    const alvo = e.target.closest("[data-acao]");
    if (!alvo) return;
    const acao = alvo.dataset.acao;
    if (acao === "escolher") {
      e.preventDefault();
      e.stopPropagation();
      if (!carregando) entrada.click();
    } else if (acao === "inicio") {
      e.preventDefault();
      pararAudio();
      mostrar("inicio");
    } else if (acao === "como-funciona") {
      e.preventDefault();
      pararAudio();
      mostrar("inicio");
      $("#como-funciona").scrollIntoView({ behavior: "smooth" });
    }
  });

  // ---------- envio ----------
  dropzone.addEventListener("click", (e) => {
    if (e.target.closest("[data-acao]")) return;   // o botão de dentro já abre o seletor (abrir 2× perde a foto no iPhone)
    if (!carregando) entrada.click();
  });
  dropzone.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !carregando) { e.preventDefault(); entrada.click(); }
  });
  ["dragenter", "dragover"].forEach((t) => dropzone.addEventListener(t, (e) => {
    e.preventDefault(); dropzone.classList.add("arrastando");
  }));
  ["dragleave", "drop"].forEach((t) => dropzone.addEventListener(t, (e) => {
    e.preventDefault(); dropzone.classList.remove("arrastando");
  }));
  dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) enviar(f);
  });
  // fora do card, soltar um arquivo não faz o navegador sair do site para abrir a imagem
  ["dragover", "drop"].forEach((t) => window.addEventListener(t, (e) => e.preventDefault()));
  let ultimoArquivo = null;
  function aoEscolher() {
    const f = entrada.files && entrada.files[0];
    if (!f || f === ultimoArquivo) return;
    ultimoArquivo = f;
    enviar(f);
    setTimeout(() => { entrada.value = ""; ultimoArquivo = null; }, 0);
  }
  entrada.addEventListener("change", aoEscolher);
  entrada.addEventListener("input", aoEscolher);

  async function enviar(arquivo) {
    try {
      await enviarFoto(arquivo);
    } catch (err) {
      console.error(err);
      carregando = false;
      dropzone.classList.remove("carregando");
      falhar(`Algo deu errado ao abrir a foto (${err && err.message ? err.message : err}). Tente de novo.`);
    }
  }

  async function enviarFoto(arquivo) {
    erro.textContent = "";
    if (arquivo.type && !arquivo.type.startsWith("image/")) return falhar("Esse arquivo não é uma imagem.");

    pararAudio();
    mostrar("inicio");
    if (urlFoto) URL.revokeObjectURL(urlFoto);
    urlFoto = URL.createObjectURL(arquivo);
    const exemplo = $(".foto-exemplo");
    exemplo.src = urlFoto;
    exemplo.classList.add("com-imagem");

    carregando = true;
    dropzone.classList.add("carregando");
    const msg = $("#msg-carregando");
    msg.textContent = "Ouvindo sua foto…";
    const lento = setTimeout(() => {
      msg.textContent = "Acordando o servidor… a primeira foto do dia pode levar uns 20 segundos.";
    }, 8000);
    try {
      const dados = await recomendar(arquivo, [...escolhidos]);
      renderPlaylist(dados);
      mostrar("playlist");
    } catch (err) {
      console.error(err);
      falhar(err.message || "Não foi possível gerar a playlist. Tente de novo.");
    } finally {
      clearTimeout(lento);
      carregando = false;
      dropzone.classList.remove("carregando");
    }
  }

  function falhar(msg) {
    erro.textContent = msg;
    mostrar("inicio");
    erro.scrollIntoView({ behavior: "smooth", block: "center" });   // no celular o erro fica abaixo da dobra
  }

  // ---------- playlist ----------
  const ICONE_PLAY = '<svg viewBox="0 0 12 12"><path d="M3 1.5v9l7.5-4.5z"/></svg>';
  const ICONE_PAUSA = '<svg viewBox="0 0 12 12"><path d="M2.5 1.5h2.5v9H2.5zM7 1.5h2.5v9H7z"/></svg>';
  const ICONE_SETA = '<svg class="ico" viewBox="0 0 24 24"><path d="M7 17 17 7M8 7h9v9"/></svg>';

  const limitar = (v) => Math.max(0, Math.min(100, v));

  function descrever(d) {
    if (d.modo_festa) {
      return "Parece uma festa. Puxamos a seleção para músicas alegres e agitadas, com o clima dos rótulos abaixo.";
    }
    const humor = d.valencia_pct >= 60 ? "alegre" : d.valencia_pct <= 40 ? "melancólico" : "neutro";
    const ritmo = d.energia_pct >= 60 ? "agitado" : d.energia_pct <= 40 ? "calmo" : "moderado";
    return `A foto transmite um clima ${humor} e ${ritmo}. Buscamos músicas na mesma posição de humor e energia.`;
  }

  function renderPlaylist(d) {
    $("#foto-enviada").src = urlFoto;
    $("#pl-nome").textContent = d.nome || "Sua seleção";
    const n = d.faixas.length;
    $("#pl-meta").textContent = `${n} ${n === 1 ? "faixa" : "faixas"} · Prévias de 30 segundos`;

    $("#criterios-texto").textContent = descrever(d);
    $("#txt-valencia").textContent = `percentil ${Math.round(d.valencia_pct)}`;
    $("#txt-energia").textContent = `percentil ${Math.round(d.energia_pct)}`;
    const tags = $("#tags");
    tags.innerHTML = "";
    (d.rotulos || []).forEach((r) => {
      const t = document.createElement("span");
      t.textContent = r;
      tags.appendChild(t);
    });
    const aviso = $("#pl-aviso");
    aviso.hidden = !d.aviso;
    aviso.textContent = d.aviso || "";
    $("#bar-valencia").style.width = "0";
    $("#bar-energia").style.width = "0";
    requestAnimationFrame(() => requestAnimationFrame(() => {
      $("#bar-valencia").style.width = `${limitar(d.valencia_pct)}%`;
      $("#bar-energia").style.width = `${limitar(d.energia_pct)}%`;
    }));

    const lista = $("#faixas");
    lista.innerHTML = "";
    d.faixas.forEach((f) => {
      const linha = document.createElement("div");
      linha.className = "linha faixa";
      linha.setAttribute("role", "row");
      linha.innerHTML = `
        <button class="play" aria-label="Tocar prévia">${ICONE_PLAY}</button>
        <div class="faixa-info">
          <div class="faixa-titulo"></div>
          <div class="faixa-artista"></div>
        </div>
        <div class="previa"><div class="progresso"><div></div></div><span class="tempo">30 s</span></div>
        <a class="link-deezer" target="_blank" rel="noopener" href="https://www.deezer.com/track/${encodeURIComponent(f.deezer_id)}"><span class="txt">Ouvir no Deezer</span>${ICONE_SETA}</a>`;
      linha.querySelector(".faixa-titulo").textContent = f.titulo;
      linha.querySelector(".faixa-artista").textContent = f.genero ? `${f.artista} · ${f.genero}` : f.artista;
      if (f.previa_url) previas.set(f.deezer_id, f.previa_url);
      linha.querySelector(".play").addEventListener("click", () => alternar(linha, f.deezer_id));
      lista.appendChild(linha);
    });
  }

  // ---------- áudio ----------
  const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  function semPrevia(linha) {
    const play = linha.querySelector(".play");
    play.disabled = true;
    play.setAttribute("aria-label", "Prévia indisponível");
    linha.querySelector(".tempo").textContent = "sem prévia";
  }

  async function alternar(linha, deezerId) {
    if (linhaAtual === linha) {
      if (player.paused) player.play().catch(() => pararAudio());
      else player.pause();
      return;
    }
    pararAudio();
    linhaAtual = linha;
    linha.classList.add("tocando");
    linha.querySelector(".tempo").textContent = "carregando…";
    try {
      if (!previas.has(deezerId)) previas.set(deezerId, await buscarPrevia(deezerId));
      if (linhaAtual !== linha) return;          // a pessoa clicou em outra faixa enquanto isso
      player.src = previas.get(deezerId);
      await player.play();
    } catch {
      previas.delete(deezerId);
      if (linhaAtual === linha) pararAudio();
      semPrevia(linha);
    }
  }

  function pararAudio() {
    player.pause();
    player.removeAttribute("src");
    if (linhaAtual) {
      linhaAtual.classList.remove("tocando");
      linhaAtual.querySelector(".play").innerHTML = ICONE_PLAY;
      if (!linhaAtual.querySelector(".play").disabled) linhaAtual.querySelector(".tempo").textContent = "30 s";
      linhaAtual.querySelector(".progresso div").style.width = "0";
    }
    linhaAtual = null;
  }

  player.addEventListener("play", () => linhaAtual && (linhaAtual.querySelector(".play").innerHTML = ICONE_PAUSA));
  player.addEventListener("pause", () => linhaAtual && (linhaAtual.querySelector(".play").innerHTML = ICONE_PLAY));
  player.addEventListener("ended", pararAudio);
  player.addEventListener("timeupdate", () => {
    if (!linhaAtual) return;
    const total = player.duration || 30;
    linhaAtual.querySelector(".progresso div").style.width = `${(player.currentTime / total) * 100}%`;
    linhaAtual.querySelector(".tempo").textContent = `${mmss(player.currentTime)} / ${mmss(total)}`;
  });
})();
