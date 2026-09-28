"""Blend a tree model with the ridge baseline: pred = w * tree + (1 - w) * ridge.

The weight is tuned on time folds T1-T2 only and then checked on T3 (never tuned on it).
Run after the time-fold CV: python -m src.ensemble
"""
import numpy as np
import pandas as pd

from .config import ARTIFACTS, REPORTS

RIDGE = "B3 ridge (log rpm)"
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def blend_table(tree):
    d = pd.read_csv(ARTIFACTS / "cv_time.csv", parse_dates=["date"])
    d = d[~d.corrupted]
    w = d[d.model.isin([tree, RIDGE])].pivot_table(index=["fold", "load_id"], columns="model", values="pred")
    y = d[d.model == tree].set_index(["fold", "load_id"]).y.reindex(w.index)
    folds = w.index.get_level_values("fold")
    tune = folds.str.startswith(("T1", "T2"))

    def mae(weight, mask):
        p = weight * w[tree] + (1 - weight) * w[RIDGE]
        return (p - y).abs()[mask].groupby(folds[mask]).mean().mean()  # mean of per-fold MAEs

    best = min(GRID, key=lambda g: mae(g, tune))
    rows = []
    for name, weight in ((tree, 1.0), (RIDGE, 0.0), (f"blend w={best:.2f} (tuned on T1-T2)", best),
                         ("blend w=0.50", 0.5)):
        p = weight * w[tree] + (1 - weight) * w[RIDGE]
        err = (p - y).abs().groupby(folds).mean()
        rows.append({"model": name, **err.round(1).to_dict(), "mean": round(err.mean(), 1), "worst": round(err.max(), 1)})
    return best, pd.DataFrame(rows)


def main(tree="A34 A27 + recency half-life 60d"):
    best, t = blend_table(tree)
    md = [f"# Ensemble check: {tree} + ridge", "", f"Weight on the tree tuned on T1-T2: {best:.2f}", "",
          t.to_markdown(index=False), ""]
    (REPORTS / "ensemble_check.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
