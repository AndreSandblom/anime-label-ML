import pandas as pd
import numpy as np

# loading
def load_anime(anime_csv: str) -> pd.DataFrame:
    df = pd.read_csv(anime_csv)
    print("Loaded shape:", df.shape)
    print("Columns:", list(df.columns))
    return df


# feature encoding (into numeric values)
# genre encoding
def encode_genres_multi_hot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns 'genre' column (comma-separated) into many 0/1 columns.
    Example: genre__Action, genre__Comedy, ...
    """
    if "genre" not in df.columns:
        return df

    # collect all unique genres
    all_genres = sorted({g.strip() for s in df["genre"] for g in str(s).split(",") if g.strip()})

    # create 0/1 column per genre (each anime will have all genres labeled 0 or 1)
    for g in all_genres:
        df[f"genre__{g}"] = df["genre"].apply(lambda s: 1 if g in [x.strip() for x in str(s).split(",")] else 0)

    # drop original text column 
    return df.drop(columns=["genre"])

