import pandas as pd

anime = pd.read_csv("data/anime.csv")
seed = pd.read_csv("data/labels_seed_50.csv")
test = pd.read_csv("data/labels_test_10.csv")

ak = anime["name"].str.lower().str.strip()

for df, name in [(seed, "seed"), (test, "test")]:
    lk = df["name"].str.lower().str.strip()
    missing = df[~lk.isin(ak)]
    if missing.empty:
        print(f"{name}: all titles match the anime.csv")
    else:
        print(f"{name}: missing {len(missing)} titles:")
        print(missing["name"].tolist())
