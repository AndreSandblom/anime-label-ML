from src.full_pipeline import run_full_pipeline
from src.config import ANIME_CSV, SEED_LABELS, OUTDIR, K, ALPHA, CHUNK_SIZE

def main():
    print("Running baseline experiment...")
    print(f"ANIME_CSV   = {ANIME_CSV}")
    print(f"SEED_LABELS = {SEED_LABELS}")
    print(f"k={K}, alpha={ALPHA}, chunk_size={CHUNK_SIZE}")

    run_full_pipeline(
        anime_csv=ANIME_CSV,
        seed_labels_csv=SEED_LABELS,
        outdir=OUTDIR,
        k=K,
        alpha=ALPHA,
        chunk_size=CHUNK_SIZE,
        limit_rows=None,
    )

    print("\nNow run evaluation:")
    print("python -m src.eval_on_test")

if __name__ == "__main__":
    main()
