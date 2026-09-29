"""Tables and charts for the trials study (benchmarks/trials_study.py)."""
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).parent / "results"


def v2():
    df = pd.read_csv(R / "trials_v2.csv")
    base = df[df.model == "seasonal_naive"].set_index(["dataset", "origin"]).WIS
    ml = df[df.model != "seasonal_naive"].copy()
    common = sorted(set(ml[ml.trials == ml.trials.max()].origin))            # origins every level has
    ml = ml[ml.origin.isin(common)]
    ml["rel"] = [w / base.get((d, o)) for w, d, o in zip(ml.WIS, ml.dataset, ml.origin)]
    tab = ml.groupby(["dataset", "model", "trials"]).rel.mean().unstack("trials").round(2)
    rt = ml.groupby(["model", "trials"]).seconds.mean().unstack("trials").round(0)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (name, g) in zip(axes[:3], tab.groupby(level=0)):
        for model, row in g.droplevel(0).iterrows():
            ax.plot(row.index, row.values, marker="o", label=model)
        ax.axhline(1, color="grey", ls="--", lw=.8); ax.set_xscale("log"); ax.set_title(name)
        ax.set_xlabel("trials per model"); ax.set_ylabel("WIS vs baseline (lower = better)")
    for model, row in rt.iterrows():
        axes[3].plot(row.index, row.values, marker="o", label=model)
    axes[3].set_xscale("log"); axes[3].set_yscale("log"); axes[3].set_title("runtime per fit")
    axes[3].set_xlabel("trials per model"); axes[3].set_ylabel("seconds"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(R / "trials_v2.png", dpi=120); plt.close(fig)
    return tab, rt, common


def v1():
    p = R / "trials_v1.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    base = pd.read_csv(R / "trials_v2.csv")
    b = base[(base.model == "seasonal_naive") & (base.origin == 2019)].set_index("dataset").RMSE
    df["RMSE_vs_baseline"] = [r / b.get(d) for r, d in zip(df.RMSE, df.dataset)]
    return df[["dataset", "trials", "RMSE", "RMSE_vs_baseline", "v1_validation_rmse", "seconds", "selected_lags"]]


if __name__ == "__main__":
    tab, rt, common = v2()
    print("v2 WIS relative to seasonal naive (origins", common, ")\n", tab.to_string())
    print("\nv2 mean seconds per fit\n", rt.to_string())
    t1 = v1()
    if t1 is not None:
        print("\nv1\n", t1.round(2).to_string(index=False))
