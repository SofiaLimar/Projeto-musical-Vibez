/*
 * Ponto de conexão com a API (site/api/index.py, docs/decisoes.md seção 4).
 *
 * Site e API ficam no mesmo projeto da Vercel, então API_URL = "/api".
 * Com API_URL = null, o site usa dados de exemplo (MOCK), sem servidor.
 *
 * Gêneros: GET {API_URL}/generos    -> { generos: [...] }
 * Envio:  POST {API_URL}/playlist   (multipart: "foto" + "generos" em JSON, [] = qualquer)
 * Volta:  { nome, valencia_pct, energia_pct, z, modo_festa, prob_festa, rotulos, aviso,
 *           link_playlist, faixas: [{ titulo, artista, genero, deezer_id, previa_url }] }
 *   valencia_pct / energia_pct: percentil 0–100 da foto na régua z.
 */
const API_URL = "/api";

/* Gêneros que a pessoa pode escolher. A lista fica só na API (site/api/_vibez/recomendar.py).
 * Se a API não responder (ou no modo de exemplo), o site mostra só "Qualquer". */
async function carregarGeneros() {
  if (!API_URL) return [];
  try {
    const resp = await fetch(`${API_URL}/generos`);
    if (!resp.ok) return [];
    const dados = await resp.json();
    return Array.isArray(dados.generos) ? dados.generos : [];
  } catch {
    return [];
  }
}

async function recomendar(arquivo, generos = []) {
  if (!API_URL) return mock(generos);

  const corpo = new FormData();
  const foto = await reduzir(arquivo);
  if (!foto) throw new Error("Não consegui abrir essa imagem. Tente outra foto ou um print dela.");
  if (foto.size > 4 * 1024 * 1024) throw new Error("A imagem é grande demais. Tente outra foto.");
  corpo.append("foto", foto, "foto.jpg");
  corpo.append("generos", JSON.stringify(generos));
  let resp;
  try {
    resp = await fetch(`${API_URL}/playlist`, { method: "POST", body: corpo });
  } catch {
    throw new Error("Não consegui falar com o servidor. Confira a conexão e tente de novo.");
  }
  if (!resp.ok) {
    const corpoErro = await resp.json().catch(() => null);
    throw new Error((corpoErro && corpoErro.detail) || `Erro ${resp.status} ao gerar a playlist.`);
  }
  return resp.json();
}

/* A Vercel recusa envios acima de 4,5 MB e o CLIP só enxerga 224 px:
 * reduzimos a foto para no máximo 1024 px antes de mandar (fotos de celular têm 5–15 MB).
 * createImageBitmap já aplica a rotação do EXIF; se falhar, tenta via <img>
 * (Safari abre HEIC do iPhone). Devolve null se o navegador não conseguir abrir a imagem. */
function abrirImagem(arquivo) {
  return new Promise((ok, falha) => {
    const img = new Image();
    const url = URL.createObjectURL(arquivo);
    img.onload = () => { URL.revokeObjectURL(url); ok(img); };
    img.onerror = () => { URL.revokeObjectURL(url); falha(new Error("imagem ilegível")); };
    img.src = url;
  });
}

async function reduzir(arquivo, lado = 1024) {
  try {
    let bmp;
    try {
      bmp = await Promise.race([
        createImageBitmap(arquivo, { imageOrientation: "from-image" }),
        new Promise((_, falha) => setTimeout(() => falha(new Error("demorou")), 8000)),
      ]);
    } catch {
      bmp = await abrirImagem(arquivo);
    }
    const k = Math.min(1, lado / Math.max(bmp.width, bmp.height));
    const tela = document.createElement("canvas");
    tela.width = Math.round(bmp.width * k);
    tela.height = Math.round(bmp.height * k);
    tela.getContext("2d").drawImage(bmp, 0, 0, tela.width, tela.height);
    if (bmp.close) bmp.close();
    return await new Promise((ok, falha) =>
      tela.toBlob((b) => (b ? ok(b) : falha()), "image/jpeg", 0.92));
  } catch {
    return null;
  }
}

/* Acorda a API assim que a página abre: ela já baixa e carrega o CLIP. */
function acordarServidor() {
  if (API_URL) fetch(`${API_URL}/saude`).catch(() => {});
}

/* Prévia de 30 s: os links do Deezer expiram, então pedimos na hora.
 * A API do Deezer não libera CORS; por isso JSONP (docs/decisoes.md seção 4). */
function buscarPrevia(deezerId) {
  return new Promise((resolve, reject) => {
    const cb = `_deezer_${deezerId}_${Date.now()}`;
    const s = document.createElement("script");
    const limpar = () => { delete window[cb]; s.remove(); clearTimeout(t); };
    const t = setTimeout(() => { limpar(); reject(new Error("tempo esgotado")); }, 8000);
    window[cb] = (dados) => {
      limpar();
      dados && dados.preview ? resolve(dados.preview) : reject(new Error("sem prévia"));
    };
    s.onerror = () => { limpar(); reject(new Error("falha de rede")); };
    s.src = `https://api.deezer.com/track/${encodeURIComponent(deezerId)}?output=jsonp&callback=${cb}`;
    document.head.appendChild(s);
  });
}

/* Dados de exemplo (faixas reais da base). */
async function mock(generos) {
  await new Promise((r) => setTimeout(r, 1400));
  return {
    nome: "Leve e ensolarada",
    valencia_pct: 72.4,
    energia_pct: 31.0,
    z: [0.59, -0.5],
    modo_festa: false,
    prob_festa: 0.03,
    rotulos: ["serene", "dreamy", "warm", "peaceful", "wistful", "soothing"],
    aviso: generos.length ? `Exemplo de aviso: completamos ${generos.join(" e ")} com Pop.` : null,
    link_playlist: null,
    faixas: [
      { titulo: "Tuyo (Narcos Theme)", artista: "Rodrigo Amarante", genero: "Trilhas sonoras", deezer_id: 573659482, previa_url: null },
      { titulo: "Sister", artista: "Sufjan Stevens", genero: "Alternativo", deezer_id: 2725664341, previa_url: null },
      { titulo: "Wheel", artista: "John Mayer", genero: "Pop", deezer_id: 70196432, previa_url: null },
      { titulo: "Seeing You Off The Edges", artista: "Eluvium", genero: "Alternativo", deezer_id: 62743414, previa_url: null },
      { titulo: "Leaves Falling", artista: "Lullatone", genero: "Pop", deezer_id: 2100441, previa_url: null },
      { titulo: "Love Stood Still", artista: "MYMP", genero: "Pop", deezer_id: 2399964855, previa_url: null },
    ],
  };
}
