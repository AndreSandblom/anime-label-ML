from src.preprocess import load_anime

# test loading
df = load_anime("data/anime.csv")
print("Rows:", len(df))
print("Columns:", list(df.columns))
print(df.head(3))

# test encoding 
from src.preprocess import load_anime, encode_genres_multi_hot

df = load_anime("data/anime.csv")
df2 = encode_genres_multi_hot(df)

print("After genre encoding shape:", df2.shape)
print("Some genre columns:", [c for c in df2.columns if c.startswith("genre__")][:10])
print(df2.head(2))


