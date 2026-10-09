"""Paper figures from saved results (read-only). Output: figures/*.png (300 dpi) and *.pdf.

Colours: the dataviz reference categorical palette in fixed order (validated set); Fed-MediShield is always
slot 1 (blue), FedAvg always slot 2 (orange), so colour follows the method in every figure.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGS = Path(__file__).resolve().parent
OUT = LOGS.parent / "figures"
OUT.mkdir(exist_ok=True)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METHOD_COLOR = {"Fed-MediShield": SLOTS[0], "FedAvg": SLOTS[1], "Krum": SLOTS[2], "Median": SLOTS[3],
                "Trimmed mean": SLOTS[4], "RFA": SLOTS[5], "Fed-ADH": SLOTS[6]}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
CLASSES = ["Benign", "DDoS", "DoS", "Recon", "MQTT", "Spoofing"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.facecolor": "white", "figure.facecolor": "white"})


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", name)


def metrics(path):
    return pd.read_csv(path).set_index("Round")


def band(ax, frames, column, color, label):
    stack = np.vstack([f[column].to_numpy(dtype=float) for f in frames])
    rounds = frames[0].index.to_numpy()
    ax.fill_between(rounds, stack.min(0), stack.max(0), color=color, alpha=0.15, linewidth=0)
    ax.plot(rounds, stack.mean(0), color=color, linewidth=1.6, marker="o", markersize=3.5, label=label)


# Fig 1. ASR per round, Phase 2e (bsc_6 = bsc_7 on these runs), conclusive seeds 71-75.
seeds = [71, 72, 73, 74, 75]
fa = [metrics(LOGS / f"phase2e_fedavg/seed_{s}/fedavg_attacked/metrics.csv") for s in seeds]
ms = [metrics(LOGS / f"phase2e_medishield/seed_{s}/medishield_attacked/metrics.csv") for s in seeds]
fig, ax = plt.subplots(figsize=(4.6, 2.9))
band(ax, fa, "ASR", METHOD_COLOR["FedAvg"], "FedAvg")
band(ax, ms, "ASR", METHOD_COLOR["Fed-MediShield"], "Fed-MediShield")
ax.set(xlabel="Federated round", ylabel="Attack success rate (%)", ylim=(-3, 103), xticks=range(1, 16, 2))
ax.set_title("Backdoor ASR per round\n5 fresh seeds (line = mean, band = range)", fontsize=9, color=INK, loc="left")
ax.legend(loc="center right")
save(fig, "fig1_asr_per_round")

# Fig 2. Method comparison (seeds 63-67, conclusive 63/64/66/67): two panels, one measure each.
summary = pd.read_csv(LOGS / "comparison_bsc6_phase2d_seeds/per_seed.csv")
summary = summary[summary.Conclusive]
order = ["Fed-MediShield", "FedAvg", "Krum", "Median", "Trimmed mean", "RFA", "Fed-ADH"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, column, label in ((axes[0], "Final_ASR", "Final attack success rate (%)"), (axes[1], "Clean_run_F1", "Clean-run macro-F1 (%)")):
    stats = summary.groupby("Method")[column].agg(["mean", "std"]).reindex(order)
    y = np.arange(len(order))[::-1]
    ax.barh(y, stats["mean"], xerr=stats["std"], color=[METHOD_COLOR[m] for m in order], height=0.62,
            edgecolor="white", linewidth=1.5, error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    for yi, value, sd in zip(y, stats["mean"], stats["std"].fillna(0)):
        ax.text(value + sd + (2 if column == "Final_ASR" else 0.4), yi, f"{value:.2f}" if value < 1 else f"{value:.1f}",
                va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, order if ax is axes[0] else [""] * len(order))
    ax.set_xlabel(label)
    ax.grid(axis="y", visible=False)
axes[0].set_xlim(0, 118)
axes[1].set_xlim(55, 82)
fig.suptitle("Phase 2d attack, 4 conclusive seeds (mean ± SD)", fontsize=9, color=INK, x=0.02, ha="left")
save(fig, "fig2_method_comparison")

# Fig 3. Sensitivity sweeps (seeds 79-81): final ASR, MediShield vs FedAvg; small multiples.
def sweep(levels, name_for):
    rows = []
    for level in levels:
        for seed in (79, 80, 81):
            f = LOGS / f"{name_for(level)}_fedavg/seed_{seed}/fedavg_attacked/metrics.csv"
            m_candidates = [LOGS / f"{name_for(level).replace('stage3b', 'stage3c')}_medishield/seed_{seed}/medishield_attacked",
                            LOGS / f"{name_for(level)}_medishield/seed_{seed}/medishield_attacked",
                            LOGS / f"{name_for(level)}_rest_medishield/seed_{seed}/medishield_attacked"]
            m_dir = next((p for p in m_candidates if (p / "metrics.csv").exists()), None)
            stopped = m_dir is not None and (m_dir / "detector_stop.json").exists()
            rows.append({"Level": level, "Seed": seed, "FedAvg": pd.read_csv(f).ASR.iloc[-1] if f.exists() else np.nan,
                         "MediShield": np.nan if (m_dir is None or stopped) else pd.read_csv(m_dir / "metrics.csv").ASR.iloc[-1],
                         "Stopped": stopped})
    return pd.DataFrame(rows)

ratio = sweep([0.125, 0.25, 0.375, 0.5], lambda v: f"stage3b_ratio_{str(v).replace('.', 'p')}")
poison = sweep([0.2, 0.35, 0.5, 0.8], lambda v: "stage3b_ratio_0p25" if v == 0.5 else f"stage3b_poison_{str(v).replace('.', 'p')}")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, table, xlabel, ticks in ((axes[0], ratio, "Attackers out of 8", ["1", "2", "3", "4"]),
                                 (axes[1], poison, "Poison rate (fraction of attacker rows)", ["0.2", "0.35", "0.5", "0.8"])):
    x = np.arange(len(table.Level.unique()))
    for column, offset in (("FedAvg", -0.08), ("MediShield", 0.08)):
        g = table.groupby("Level")[column]
        color = METHOD_COLOR["FedAvg" if column == "FedAvg" else "Fed-MediShield"]
        complete = table.groupby("Level")["Stopped"].sum().to_numpy() == 0
        means = g.mean().to_numpy() if column == "FedAvg" else np.where(complete, g.mean().to_numpy(), np.nan)
        ax.plot(x + offset, means, color=color, linewidth=1.6, marker="o", markersize=4.5,
                label="FedAvg" if column == "FedAvg" else "Fed-MediShield")
        for xi, level in zip(x, sorted(table.Level.unique())):
            vals = table[table.Level == level][column].dropna()
            ax.scatter(np.full(len(vals), xi + offset), vals, s=9, color=color, alpha=0.45, linewidth=0)
    for xi, level in zip(x, sorted(table.Level.unique())):
        n_stop = int(table[(table.Level == level)].Stopped.sum())
        if n_stop:
            done = table[(table.Level == level) & ~table.Stopped].MediShield.dropna()
            ax.scatter(np.full(len(done), xi + 0.08), done, s=26, facecolor="white",
                       edgecolor=METHOD_COLOR["Fed-MediShield"], linewidth=1.2, zorder=3)
            ax.annotate(f"Fed-MediShield:\n{n_stop}/3 runs stopped,\n{len(done)} completed (hollow)", (xi - 0.12, 38),
                        ha="right", fontsize=7, color=INK2)
    ax.set_xticks(x, ticks)
    ax.set_xlabel(xlabel)
axes[0].set_ylabel("Final attack success rate (%)")
axes[0].set_ylim(-4, 104)
axes[0].legend(loc="center left")
fig.suptitle("Sensitivity (seeds 79–81; line = mean, dots = seeds)", fontsize=9, color=INK, x=0.02, ha="left")
save(fig, "fig3_sensitivity_sweeps")

# Fig 4. Persistence: attack in rounds 1-7 only.
pseeds = [43, 46, 88, 89]  # seed 87 excluded: FedAvg did not learn the backdoor
fa = [metrics(LOGS / f"persist_fedavg/seed_{s}/fedavg_attacked/metrics.csv") for s in pseeds]
ms = [metrics(LOGS / f"{'persist_medishield_dev' if s in (43, 46) else 'persist_medishield_fresh'}/seed_{s}/medishield_attacked/metrics.csv") for s in pseeds]
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.axvspan(0.5, 7.5, color="#f1f0ec", zorder=0)
ax.text(4, 52, "attack active", ha="center", fontsize=8, color=INK2)
ax.text(11.5, 52, "all clients honest", ha="center", fontsize=8, color=INK2)
band(ax, fa, "ASR", METHOD_COLOR["FedAvg"], "FedAvg")
band(ax, ms, "ASR", METHOD_COLOR["Fed-MediShield"], "Fed-MediShield")
ax.set(xlabel="Federated round", ylabel="Attack success rate (%)", ylim=(-3, 103), xticks=range(1, 16, 2))
ax.set_title("Backdoor persistence after the attack stops (4 seeds)", fontsize=9, color=INK, loc="left")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.72))
save(fig, "fig4_persistence")

# Fig 5. Hyperparameter sensitivity: final ASR per arm (two dev seeds).
hp = pd.read_csv(LOGS / "hparams_table/hparams_per_seed.csv")
hp["Arm"] = hp.Parameter.str.replace("_", " ") + " = " + hp.Value.astype(str)
hp.loc[hp.Parameter == "default", "Arm"] = "defaults"
arms = list(dict.fromkeys(hp.Arm))
fig, ax = plt.subplots(figsize=(5.2, 4.2))
y = np.arange(len(arms))[::-1]
for yi, arm in zip(y, arms):
    vals = hp[hp.Arm == arm].Final_ASR.to_numpy(dtype=float)
    fail = np.nanmax(vals) > 10
    ax.plot([np.nanmin(vals), np.nanmax(vals)], [yi, yi], color=GRID, linewidth=1.2, zorder=1)
    ax.scatter(vals, np.full(len(vals), yi), s=24, zorder=2, color=SLOTS[7] if fail else METHOD_COLOR["Fed-MediShield"],
               edgecolor="white", linewidth=0.8)
ax.set_yticks(y, arms, fontsize=7.5)
ax.set_xlabel("Final attack success rate (%), seeds 43 and 46")
ax.set_xlim(-3, 103)
ax.grid(axis="y", visible=False)
ax.set_title("One-at-a-time hyperparameter sensitivity\n(red = backdoor survives)", fontsize=9, color=INK, loc="left")
save(fig, "fig5_hyperparameters")

# Fig 6. Confusion matrices (row-normalised recall %), clean runs, seed 71: FedAvg vs Fed-MediShield.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
for ax, (title, path) in zip(axes, (("FedAvg, no attack", "phase2e_fedavg/seed_71/fedavg_clean/confusion_matrix.csv"),
                                    ("Fed-MediShield, under attack", "phase2e_medishield/seed_71/medishield_attacked/confusion_matrix.csv"))):
    raw = pd.read_csv(LOGS / path, header=None)
    cm = (raw.iloc[1:, 1:] if str(raw.iloc[0, 0]) == "True_Class" else raw).to_numpy(dtype=float)  # two saved formats
    pct = 100 * cm / cm.sum(1, keepdims=True)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if pct[i, j] > 55 else INK)
    ax.set_xticks(range(6), CLASSES, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(6), CLASSES if ax is axes[0] else [""] * 6, fontsize=7.5)
    ax.set_xlabel("Predicted class")
    ax.grid(False)
    ax.set_title(title, fontsize=9, color=INK)
axes[0].set_ylabel("True class")
fig.suptitle("Validation confusion matrices (untriggered data), final round, seed 71 (row %)", fontsize=9, color=INK, x=0.02, ha="left")
save(fig, "fig6_confusion_matrices")

# Table: non-IID and scalability.
ni = []
for tag, label in (("a10", "alpha = 1.0, 8 clients"), ("c16", "alpha = 0.5, 16 clients")):
    for seed in (90, 91, 92):
        f = metrics(LOGS / f"ni_{tag}_fedavg/seed_{seed}/fedavg_attacked/metrics.csv")
        m = metrics(LOGS / f"ni_{tag}_medishield/seed_{seed}/medishield_attacked/metrics.csv")
        s = json.loads((LOGS / f"ni_{tag}_medishield/seed_{seed}/medishield_attacked/summary.json").read_text())
        ni.append({"Setting": label, "Seed": seed, "FedAvg_final_ASR": f.ASR.iloc[-1], "MediShield_final_ASR": m.ASR.iloc[-1],
                   "Detection_%": s["Overall_Detection_Rate"], "Honest_FPR_%": s["Overall_Update_FPR"]})
pd.DataFrame(ni).round(2).to_csv(OUT / "table_noniid_scalability.csv", index=False)
print("wrote table_noniid_scalability.csv (alpha = 0.1: gate failed, see addendum A26)")
