from src.full_pipeline import run_full_pipeline
from src.config import ANIME_CSV, SEED_LABELS, OUTDIR, K, ALPHA, CHUNK_SIZE

def main():
    run_full_pipeline(
        anime_csv=ANIME_CSV,
        seed_labels_csv=SEED_LABELS,
        outdir=OUTDIR,
        k=K,
        alpha=ALPHA,
        chunk_size=CHUNK_SIZE,
        limit_rows=None,
    )

if __name__ == "__main__":
    main()

