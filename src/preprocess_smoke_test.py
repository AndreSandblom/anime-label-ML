from src.preprocess import load_anime
from src.preprocess import load_anime, encode_genres_multi_hot
from src.preprocess import load_anime, encode_genres_multi_hot, encode_type_one_hot

# test loading
df = load_anime("data/anime.csv")
print("Rows:", len(df))
print("Columns:", list(df.columns))
print(df.head(3))

# test genre encoding 
df = load_anime("data/anime.csv")
df2 = encode_genres_multi_hot(df)

print("After genre encoding shape:", df2.shape)
print("Some genre columns:", [c for c in df2.columns if c.startswith("genre__")][:10])
print(df2.head(2))

# test type encoding 
df = load_anime("data/anime.csv")
df = encode_genres_multi_hot(df)

print("After genre encoding shape:", df.shape)

df = encode_type_one_hot(df)

print("After type encoding shape:", df.shape)
print("Type columns:", [c for c in df.columns if c.startswith("type_")])
print(df.head(2))

from src.preprocess import build_features

names, X = build_features("data/anime.csv")

print("Names head:")
print(names.head(3))

print("\nX shape:", X.shape)
print("Any NaN?", X.isna().any().any())

# test types: should be numeric
print("\nDtypes (first 10):")
print(X.dtypes.head(10))

# test normalization roughly: means ~0, stds ~1 for numeric columns
for col in ["episodes", "rating", "members"]:
    if col in X.columns:
        print(f"\n{col}: mean={X[col].mean():.3f}, std={X[col].std():.3f}")




