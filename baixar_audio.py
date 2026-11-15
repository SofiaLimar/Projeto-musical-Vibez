"""
baixar_audio.py — passo 3 do projeto playlist a partir de foto.

Baixa as prévias de 30 s do Deezer para as músicas da coleta.
Entrada: data/lista_rotulos_1000.csv (de contar_rotulos.py), usando as N primeiras
músicas de cada rótulo (--teto, padrão 100), sem repetir música.

Para cada música:
  1. busca no Deezer em 2 etapas: "artista título"; se falhar, track:"título" (50 resultados);
  2. confere artista e título após normalizar (minúsculas, sem acento, sem parênteses,
     sem "feat.", sem pontuação; também sem espaços); aceita igualdade ou um contido no outro;
  3. pega a data de lançamento em /track/{id} (pode ser de remasterização);
  4. baixa a prévia na hora (os links do Deezer expiram);
  5. checa integridade decodificando o MP3 inteiro (soundfile).

Processa várias músicas ao mesmo tempo (--trabalhadores, padrão 4), com um controle de
velocidade compartilhado: no máximo 8 requisições/s à API (limite do Deezer: 50 a cada 5 s).
Retoma de onde parou. Falhas de conexão não são gravadas e são tentadas de novo.

Uso:
  python baixar_audio.py --amostra 50     # teste
  python baixar_audio.py                  # todas
"""
import argparse, csv, hashlib, re, threading, time, unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import soundfile as sf

DEEZER = "https://api.deezer.com"
INTERVALO = 0.125               # 8 requisições/s no total, somando todos os trabalhadores
PASTA_AUDIO = Path("audio_previas")
CAMPOS = ["id", "artista", "titulo", "rotulos", "status", "deezer_id", "deezer_artista",
          "deezer_titulo", "deezer_album", "data_lancamento", "duracao_previa", "arquivo"]

_trava = threading.Lock()
_ultimo = [0.0]
_avisos_limite = [0]


def esperar_vez():
    """Garante o intervalo mínimo entre requisições à API, somando todos os trabalhadores."""
    with _trava:
        espera = _ultimo[0] + INTERVALO - time.monotonic()
        if espera > 0:
            time.sleep(espera)
        _ultimo[0] = time.monotonic()


# ---------- normalização e conferência de nomes ----------

def normalizar(texto):
    original = str(texto).casefold().strip()
    t = unicodedata.normalize("NFKD", original).encode("ascii", "ignore").decode()
    t = re.sub(r"[\(\[].*?[\)\]]", " ", t)              # (Remastered 2011), [Live]
    t = re.sub(r"\s+-\s+.*$", " ", t)                    # " - Remastered", " - Live"
    t = re.sub(r"\b(feat|ft|featuring)\b.*$", " ", t)    # feat. Fulano
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or original                                 # nomes não latinos: mantém o original


def confere(a, b):
    if a == b or a.replace(" ", "") == b.replace(" ", ""):    # famouslastwords = famous last words
        return True
    return len(a) >= 3 and len(b) >= 3 and (a in b or b in a)


# ---------- acesso ao Deezer ----------

def get_json(url, params=None):
    for tentativa in range(5):
        esperar_vez()
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                d = r.json()
                if isinstance(d, dict) and "error" in d:
                    if d["error"].get("code") == 4:      # limite de requisições
                        with _trava:
                            _avisos_limite[0] += 1
                        time.sleep(5 * (tentativa + 1))
                        continue
                    return {}
                return d
        except (requests.RequestException, ValueError):
            pass
        time.sleep(5 * (tentativa + 1))
    return None                                          # falha de conexão


def escolher(candidatos, na, nt):
    bons = [c for c in candidatos
            if confere(na, normalizar(c["artist"]["name"])) and confere(nt, normalizar(c["title"]))]
    if not bons:
        return None, None
    bons.sort(key=lambda c: (normalizar(c["title"]) != nt, normalizar(c["artist"]["name"]) != na))
    com_previa = [c for c in bons if c.get("preview")]
    return ("ok", com_previa[0]) if com_previa else ("sem_preview", bons[0])


def buscar(artista, titulo):
    """Etapa 1: busca simples "artista título".
    Etapa 2 (se a 1 falhar): track:"título" com 50 resultados, filtrando pelo artista.
    Motivo: no teste, "CHVRCHES Asking For A Friend" voltou vazio, mas
    track:"asking for a friend" trouxe a música. O filtro artist: dá resultados sem sentido."""
    na, nt = normalizar(artista), normalizar(titulo)
    titulo_busca = nt.replace('"', "")
    tentativas = [{"q": f"{na} {nt}", "limit": 10},
                  {"q": f'track:"{titulo_busca}"', "limit": 50}]
    primeiro, sem_previa = None, None
    for params in tentativas:
        d = get_json(f"{DEEZER}/search", params)
        if d is None:
            return "erro_busca", None
        candidatos = d.get("data", [])
        if candidatos and primeiro is None:
            primeiro = candidatos[0]
        status, c = escolher(candidatos, na, nt)
        if status == "ok":
            return "ok", c
        if status == "sem_preview" and sem_previa is None:
            sem_previa = c
    if sem_previa:
        return "sem_preview", sem_previa
    if primeiro:
        return "nome_nao_confere", primeiro
    return "nao_encontrada", None


def baixar(url, destino):
    """Download do arquivo (servidor de arquivos do Deezer, não a API: fora do controle de velocidade)."""
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200 or len(r.content) < 10_000:
            return False
        destino.write_bytes(r.content)
        return True
    except requests.RequestException:
        return False


def checar(arquivo):
    """Decodifica o áudio inteiro com soundfile (a mesma biblioteca que o librosa usa
    para ler arquivos). Falha se qualquer trecho do MP3 estiver corrompido."""
    try:
        dados, taxa = sf.read(str(arquivo))
        return len(dados) / taxa if len(dados) else None
    except Exception:
        return None


# ---------- uma música ----------

def processar(m):
    linha = dict(id=m.id, artista=m.artista, titulo=m.titulo, rotulos=m.rotulos)
    status, c = buscar(m.artista, m.titulo)
    if c:
        linha.update(deezer_id=c["id"], deezer_artista=c["artist"]["name"],
                     deezer_titulo=c["title"],
                     deezer_album=c.get("album", {}).get("title", ""))
    if status == "ok":
        info = get_json(f"{DEEZER}/track/{c['id']}") or {}
        linha["data_lancamento"] = info.get("release_date", "")
        arq = PASTA_AUDIO / f"{m.id}.mp3"
        if not arq.exists() and not baixar(c["preview"], arq):
            status = "erro_download"
        else:
            duracao = checar(arq)
            if duracao is None:
                status = "arquivo_invalido"
                arq.unlink(missing_ok=True)
            else:
                linha.update(duracao_previa=round(duracao, 1), arquivo=str(arq))
    linha["status"] = status
    return linha


# ---------- preparação da lista ----------

def preparar(entrada, teto):
    d = pd.read_csv(entrada).fillna({"artista": "", "titulo": ""})
    d["ordem"] = d.groupby("rotulo").cumcount() + 1
    d = d[d["ordem"] <= teto].copy()
    d["chave"] = d["artista"].map(normalizar) + " — " + d["titulo"].map(normalizar)
    g = (d.groupby("chave")
           .agg(artista=("artista", "first"), titulo=("titulo", "first"),
                rotulos=("rotulo", lambda s: "|".join(sorted(set(s)))))
           .reset_index())
    g["id"] = g["chave"].map(lambda k: hashlib.sha1(k.encode()).hexdigest()[:12])
    return g


# ---------- programa principal ----------

def main(entrada, teto, amostra, trabalhadores):
    PASTA_AUDIO.mkdir(exist_ok=True)
    saida = Path(f"data/audio_teto{teto}.csv")
    musicas = preparar(entrada, teto)
    feitos = set(pd.read_csv(saida)["id"]) if saida.exists() else set()
    fila = musicas[~musicas["id"].isin(feitos)].sample(frac=1, random_state=42)
    if amostra:
        fila = fila.head(amostra)
    print(f"{len(musicas)} músicas únicas | já feitas: {len(feitos)} | "
          f"nesta rodada: {len(fila)} | trabalhadores: {trabalhadores}")

    novo = not saida.exists()
    inicio, nao_gravadas = time.time(), 0
    with open(saida, "a", newline="", encoding="utf-8") as f, \
         ThreadPoolExecutor(max_workers=trabalhadores) as ex:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if novo:
            w.writeheader()
        tarefas = [ex.submit(processar, m) for m in fila.itertuples()]
        for i, t in enumerate(as_completed(tarefas), 1):
            linha = t.result()
            print(f"[{i}/{len(fila)}] {linha['status']}: {linha['artista']} — {linha['titulo']}")
            if linha["status"] in ("erro_busca", "erro_download"):
                nao_gravadas += 1                         # será tentada de novo na próxima rodada
                continue
            w.writerow(linha)                             # só o programa principal escreve no CSV
            f.flush()

    tempo = time.time() - inicio
    res = pd.read_csv(saida)
    restantes = len(musicas) - len(res)
    tamanho = sum(p.stat().st_size for p in PASTA_AUDIO.glob("*.mp3"))
    n_mp3 = len(list(PASTA_AUDIO.glob("*.mp3")))
    print("\n=== Resumo ===")
    print(res["status"].value_counts().to_string())
    print(f"ok: {(res['status'] == 'ok').mean():.0%} das {len(res)} processadas")
    print(f"falhas de conexão (serão tentadas de novo): {nao_gravadas}")
    print(f"vezes que o Deezer pediu para esperar: {_avisos_limite[0]}")
    print(f"tempo desta rodada: {tempo/60:.1f} min | {tempo/max(len(fila),1):.2f} s por música")
    if n_mp3:
        print(f"áudio: {n_mp3} arquivos, {tamanho/1e6:.0f} MB ({tamanho/n_mp3/1e6:.2f} MB por prévia)")
    if restantes > 0 and len(fila):
        print(f"faltam {restantes} músicas — estimativa: {restantes * tempo/len(fila)/3600:.1f} h")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--entrada", default="data/lista_rotulos_1000.csv")
    p.add_argument("--teto", type=int, default=100)
    p.add_argument("--amostra", type=int, default=0, help="0 = todas")
    p.add_argument("--trabalhadores", type=int, default=4)
    a = p.parse_args()
    main(a.entrada, a.teto, a.amostra, a.trabalhadores)