from src.preprocess import load_anime

df = load_anime("data/anime.csv")
print("Rows:", len(df))
print("Columns:", list(df.columns))
print(df.head(3))
