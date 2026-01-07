import pandas as pd
import numpy as np
from src.config import TEST_LABELS

preds = pd.read_csv("outputs/predictions.csv")
test = pd.read_csv(TEST_LABELS)

m = preds.merge(test[["name", "happy_ending"]], on="name", how="inner")

acc = (m["pred"] == m["happy_ending"]).mean()
print("Test matched:", len(m))
print("Test accuracy:", round(acc, 3))

cm = np.zeros((2, 2), dtype=int)
for t, p in zip(m["happy_ending"], m["pred"]):
    cm[int(t), int(p)] += 1

print("Confusion matrix (rows=true, cols=pred):\n", cm)
print("\nDetails:\n", m[["name", "happy_ending", "pred", "confidence"]] if "confidence" in m.columns else m)
