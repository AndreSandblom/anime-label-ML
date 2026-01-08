from pathlib import Path

# find repo root dynamically (folder containing src/ and data/)
p = Path(__file__).resolve()

while p != p.parent and not ((p.parent / "src").exists() and (p.parent / "data").exists()):
    p = p.parent

REPO_ROOT = p.parent

DATA_DIR = REPO_ROOT / "data"
OUTDIR = REPO_ROOT / "outputs"

ANIME_CSV = DATA_DIR / "anime.csv"
SEED_LABELS = DATA_DIR / "labels_seed_50.csv"
TEST_LABELS = DATA_DIR / "labels_test_10.csv"

# model hyperparams (optimised)
K = 15
ALPHA = 0.9
CHUNK_SIZE = 200
