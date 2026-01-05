import pandas as pd
import numpy as np

def load_anime(anime_csv: str) -> pd.DataFrame:
    df = pd.read_csv(anime_csv)
    print("Loaded shape:", df.shape)
    print("Columns:", list(df.columns))
    return df
