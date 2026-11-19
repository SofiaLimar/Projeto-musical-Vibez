"""
montar_popularidade.py — junta popularidade e gênero numa tabela só.

- pop: média do percentil do rank da faixa e do percentil dos fãs do artista (0 a 1)
- generos: gêneros do álbum no Deezer ("|"); se o álbum não tiver, usa os gêneros mais
  comuns entre os outros álbuns do mesmo artista na base
- genero: o primeiro da lista (gênero principal)

Saída: data/popularidade_final.csv -> subir para MyDrive/mapa-sonoro/
"""
import pandas as pd

faixas = pd.read_csv("data/popularidade.csv")
fas = pd.read_csv("data/fas_artistas.csv")
gen = pd.read_csv("data/generos_albuns.csv").fillna({"generos": ""})

df = (faixas[faixas.status == "ok"]
      .merge(fas[fas.status == "ok"][["artista_deezer_id", "nb_fan"]], on="artista_deezer_id", how="left")
      .merge(gen[["album_deezer_id", "generos"]], on="album_deezer_id", how="left"))
df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
df["nb_fan"] = pd.to_numeric(df["nb_fan"], errors="coerce")
df["pop"] = pd.concat([df["rank"].rank(pct=True), df["nb_fan"].rank(pct=True)], axis=1).mean(axis=1)

# gênero: do álbum; se vazio, o mais comum do artista
df["generos"] = df["generos"].fillna("")
df["genero"] = df["generos"].str.split("|").str[0]
moda = (df[df.genero != ""].groupby("artista_deezer_id").genero
        .agg(lambda s: s.value_counts().index[0]))
vazio = df.genero == ""
df.loc[vazio, "genero"] = df.loc[vazio, "artista_deezer_id"].map(moda).fillna("")
df["genero_origem"] = "album"
df.loc[vazio & (df.genero != ""), "genero_origem"] = "artista"
df.loc[df.genero == "", "genero_origem"] = "sem"

df[["deezer_id", "rank", "nb_fan", "pop", "genero", "generos", "genero_origem"]].to_csv(
    "data/popularidade_final.csv", index=False)

print(f"{len(df)} músicas | rank mediano {df['rank'].median():.0f} | fãs medianos {df.nb_fan.median():.0f}")
print("\norigem do gênero:\n" + df.genero_origem.value_counts().to_string())
print("\ngênero principal (todas | metade mais popular):")
print(pd.DataFrame({"todas": df.genero.replace("", "(sem)").value_counts(),
                    "pop>=0.5": df[df["pop"] >= 0.5].genero.replace("", "(sem)").value_counts()}).fillna(0).astype(int).to_string())

nomes = pd.read_csv("data/audio_teto100.csv")[["deezer_id", "artista", "titulo"]].drop_duplicates("deezer_id")
v = df.merge(nomes, on="deezer_id")
for q in [0.3, 0.5, 0.7]:
    print(f"\n--- em torno do percentil {q:.0%} de pop ---")
    print(v.iloc[(v["pop"] - q).abs().argsort()[:8]][["artista", "titulo", "genero", "rank", "nb_fan", "pop"]].to_string(index=False))
