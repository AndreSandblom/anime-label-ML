import pandas as pd
import numpy as np

# load data
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

# type encoding (one-hot)
def encode_type_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'type' (TV/Movie/OVA/...) into one-hot columns:
    type_TV, type_Movie, ...
    """
    if "type" not in df.columns:
        return df

    dummies = pd.get_dummies(df["type"], prefix="type", dummy_na=True)
    df = pd.concat([df.drop(columns=["type"]), dummies], axis=1)
    return df

# normalize (make values comparable) - z-score
def zscore_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    """
    Z-score normalize numeric columns: (x - mean) / std
    Prevents large-scale columns like 'members' from dominating distances.
    """
    for c in cols:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            mu = x.mean()
            sd = x.std()
            if sd == 0 or pd.isna(sd):
                sd = 1.0
            df[c] = (x - mu) / sd
    return df

# wrapper function to build all features
def build_features(anime_csv: str):
    """
    Returns:
      names: Series of anime titles
      Xdf: DataFrame of numeric features (ready for ML distance calculations)
    """
    df = load_anime(anime_csv)
    names = df["name"].copy()

    df = encode_genres_multi_hot(df)
    df = encode_type_one_hot(df)

    # convert one-hot booleans to 0/1
    df = df.astype({c: int for c in df.columns if df[c].dtype == bool})

    # normalize numeric columns (only if they exist)
    df = zscore_numeric(df, ["episodes", "rating", "members", "year"])

    Xdf = df.drop(columns=["name"]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return names.reset_index(drop=True), Xdf.reset_index(drop=True)

