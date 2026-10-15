"""Paper figures, part 2: ToN-IoT figures, detection heatmaps, two-dataset summary, score distributions and the
method overview. Read-only. Same palette and colour assignment as make_figures.py (MediShield blue, FedAvg orange)."""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

LOGS = Path(__file__).resolve().parent
OUT = LOGS.parent / "figures"
OUT.mkdir(exist_ok=True)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
COLOR = {"Fed-MediShield": SLOTS[0], "FedAvg": SLOTS[1], "Krum": SLOTS[2], "Median": SLOTS[3], "Trimmed mean": SLOTS[4],
         "RFA": SLOTS[5], "Fed-ADH": SLOTS[6], "Oracle": "#8a8985"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#f1f0ec"
TON_CLASSES = ["Normal", "Backdoor", "DDoS", "DoS", "Injection", "MITM", "Password", "Ransomware", "Scanning", "XSS"]
CIC_CLASSES = ["Benign", "DDoS", "DoS", "Recon", "MQTT", "Spoofing"]
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


def confusion(path):
    raw = pd.read_csv(path, header=None)
    cm = (raw.iloc[1:, 1:] if str(raw.iloc[0, 0]) == "True_Class" else raw).to_numpy(dtype=float)
    return 100 * cm / cm.sum(1, keepdims=True)


TON_SEEDS = [93, 94, 95, 96, 97]
CIC_SEEDS = [71, 72, 73, 74, 75]

# Fig 7. ToN-IoT ASR per round.
fa = [metrics(LOGS / f"tond_fedavg/seed_{s}/fedavg_attacked/metrics.csv") for s in TON_SEEDS]
ms = [metrics(LOGS / f"tond_medishield_b/seed_{s}/medishield_attacked/metrics.csv") for s in TON_SEEDS]
fig, ax = plt.subplots(figsize=(4.6, 2.9))
band(ax, fa, "ASR", COLOR["FedAvg"], "FedAvg")
band(ax, ms, "ASR", COLOR["Fed-MediShield"], "Fed-MediShield")
ax.set(xlabel="Federated round", ylabel="Attack success rate (%)", ylim=(-3, 103), xticks=range(1, 16, 2))
ax.set_title("ToN-IoT: backdoor ASR per round\n5 seeds (line = mean, band = range)", fontsize=9, color=INK, loc="left")
ax.legend(loc="center right")
save(fig, "fig7_ton_asr_per_round")

# Fig 8. ToN-IoT method comparison.
table = pd.read_csv(LOGS / "comparison_ton/per_seed.csv")
order = ["Fed-MediShield", "FedAvg", "Krum", "Median", "Trimmed mean", "RFA", "Fed-ADH"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, column, label, lim in ((axes[0], "Final_ASR", "Final attack success rate (%)", (0, 118)),
                               (axes[1], "Clean_run_F1", "Clean-run macro-F1 (%)", (45, 78))):
    stats = table.groupby("Method")[column].agg(["mean", "std"]).reindex(order)
    y = np.arange(len(order))[::-1]
    ax.barh(y, stats["mean"], xerr=stats["std"], color=[COLOR[m] for m in order], height=0.62, edgecolor="white",
            linewidth=1.5, error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    for yi, value, sd in zip(y, stats["mean"], stats["std"].fillna(0)):
        ax.text(value + sd + (2 if column == "Final_ASR" else 0.4), yi, f"{value:.2f}" if value < 1 else f"{value:.1f}",
                va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, order if ax is axes[0] else [""] * len(order))
    ax.set_xlabel(label)
    ax.set_xlim(*lim)
    ax.grid(axis="y", visible=False)
fig.suptitle("ToN-IoT, 5 seeds (mean ± SD; descriptive evaluation)", fontsize=9, color=INK, x=0.02, ha="left")
save(fig, "fig8_ton_method_comparison")

# Fig 9. ToN-IoT confusion matrices (10 classes).
fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.3))
for ax, (title, path) in zip(axes, (("FedAvg, no attack", LOGS / "tond_fedavg/seed_93/fedavg_clean/confusion_matrix.csv"),
                                    ("Fed-MediShield, under attack", LOGS / "tond_medishield_b/seed_93/medishield_attacked/confusion_matrix.csv"))):
    pct = confusion(path)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(10):
        for j in range(10):
            if pct[i, j] >= 0.5:
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=6, color="white" if pct[i, j] > 55 else INK)
    ax.set_xticks(range(10), TON_CLASSES, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(10), TON_CLASSES if ax is axes[0] else [""] * 10, fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.grid(False)
    ax.set_title(title, fontsize=9, color=INK)
axes[0].set_ylabel("True class")
fig.suptitle("ToN-IoT validation confusion matrices, final round, seed 93 (row %; cells < 0.5% blank)", fontsize=9,
             color=INK, x=0.02, ha="left")
save(fig, "fig9_ton_confusion_matrices")

# Fig 10. Detection heatmaps: score / threshold per client and round; attackers marked.
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
for ax, (title, path) in zip(axes, (("CICIoMT2024, seed 71", LOGS / "phase2e_medishield/seed_71/medishield_attacked/client_decisions.csv"),
                                    ("ToN-IoT, seed 93", LOGS / "tond_medishield_b/seed_93/medishield_attacked/client_decisions.csv"))):
    d = pd.read_csv(path)
    d["Ratio"] = d.Score.clip(lower=0) / d.Alert_Threshold
    grid = d.pivot(index="Client", columns="Round", values="Ratio").to_numpy()
    image = ax.imshow(np.clip(grid, 0, 3), cmap="Blues", vmin=0, vmax=3, aspect="auto")
    for _, r in d[d.Flagged == 1].iterrows():
        ax.text(r.Round - 1, r.Client, "×", ha="center", va="center", fontsize=8, color="white" if r.Ratio > 1.6 else INK)
    for client in sorted(d[d.Is_Compromised == 1].Client.unique()):
        ax.add_patch(Rectangle((-0.5, client - 0.5), grid.shape[1], 1, fill=False, edgecolor=SLOTS[7], linewidth=1.4))
    ax.set_xticks(range(0, 15, 2), range(1, 16, 2))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_xlabel("Round")
    ax.grid(False)
    ax.set_title(title, fontsize=9, color=INK)
axes[0].set_ylabel("Client")
bar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
bar.set_label("Suspicion score / alert threshold (capped at 3)", fontsize=8)
fig.suptitle("Where Fed-MediShield raises alerts (× = flagged; red outline = attacker)", fontsize=9, color=INK, x=0.02, ha="left", y=1.06)
fig.savefig(OUT / "fig10_detection_heatmaps.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig10_detection_heatmaps.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote fig10_detection_heatmaps")

# Fig 11. Two-dataset summary: final ASR and attacked-run F1 relative to clean FedAvg, MediShield vs FedAvg vs oracle.
def collect(seeds, ms_root, fa_root, or_root):
    out = []
    for s in seeds:
        fc = pd.read_csv(LOGS / f"{fa_root}/seed_{s}/fedavg_clean/metrics.csv").Clean_F1.iloc[-1]
        for method, path in (("FedAvg", f"{fa_root}/seed_{s}/fedavg_attacked"), ("Fed-MediShield", f"{ms_root}/seed_{s}/medishield_attacked"),
                             ("Oracle", f"{or_root}/seed_{s}/oracle_residual_attacked")):
            m = pd.read_csv(LOGS / path / "metrics.csv").iloc[-1]
            out.append({"Seed": s, "Method": method, "ASR": m.ASR, "dF1": m.Clean_F1 - fc})
    return pd.DataFrame(out)

summary = {"CICIoMT2024": collect(CIC_SEEDS, "phase2e_medishield", "phase2e_fedavg", "phase2e_oracle"),
           "ToN-IoT": collect(TON_SEEDS, "tond_medishield_b", "tond_fedavg", "tond_oracle")}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
methods = ["FedAvg", "Fed-MediShield", "Oracle"]
for ax, column, label in ((axes[0], "ASR", "Final attack success rate (%)"), (axes[1], "dF1", "Attacked-run F1 minus clean FedAvg F1 (pp)")):
    x = np.arange(2)
    for k, method in enumerate(methods):
        means = [summary[d][summary[d].Method == method][column].mean() for d in summary]
        sds = [summary[d][summary[d].Method == method][column].std() for d in summary]
        ax.bar(x + (k - 1) * 0.26, means, width=0.24, yerr=sds, color=COLOR[method], edgecolor="white", linewidth=1.5,
               error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2}, label=method)
        if column == "ASR":
            for xi, v in zip(x, means):
                ax.text(xi + (k - 1) * 0.26, v + 4, f"{v:.2f}" if v < 1 else f"{v:.1f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, list(summary))
    ax.set_ylabel(label, fontsize=8)
    ax.grid(axis="x", visible=False)
    if column == "dF1":
        ax.axhline(0, color=INK2, linewidth=0.8)
axes[0].set_ylim(0, 118)
axes[0].legend(loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Both datasets, 5 seeds each (mean ± SD): backdoor removed; utility equals perfect detection",
             fontsize=9, color=INK, x=0.02, ha="left")
save(fig, "fig11_two_dataset_summary")

# Fig 12. Score distributions: honest vs poisoned updates, relative to the threshold.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, (title, pattern) in zip(axes, (("CICIoMT2024 (5 seeds)", "phase2e_medishield/seed_*/medishield_attacked/client_decisions.csv"),
                                       ("ToN-IoT (5 seeds)", "tond_medishield_b/seed_*/medishield_attacked/client_decisions.csv"))):
    d = pd.concat(pd.read_csv(p) for p in glob.glob(str(LOGS / pattern)))
    ratio = d.Score.clip(lower=0) / d.Alert_Threshold
    bins = np.linspace(0, 4, 41)
    for flag, label, color in ((0, "honest updates", COLOR["FedAvg"]), (1, "poisoned updates", COLOR["Fed-MediShield"])):
        values = np.clip(ratio[d.Is_Poisoned_Update == flag], 0, 4)
        ax.hist(values, bins=bins, color=color, alpha=0.75, label=f"{label} (n={len(values)})",
                weights=np.full(len(values), 100 / max(len(values), 1)), edgecolor="white", linewidth=0.5)
    ax.axvline(1.0, color=INK, linewidth=1, linestyle="--")
    ax.text(1.04, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] else 1, "threshold", fontsize=7, color=INK2)
    ax.set_xlabel("Suspicion score / alert threshold (capped at 4)")
    ax.set_title(title, fontsize=9, color=INK)
    ax.legend(fontsize=7, loc="upper right")
axes[0].set_ylabel("Share of updates (%)")
save(fig, "fig12_score_distributions")

# Fig 0. Method overview diagram.
fig, ax = plt.subplots(figsize=(7.0, 2.35))
ax.set_xlim(0, 100); ax.set_ylim(-8, 31); ax.axis("off")
steps = [(0.5, "Clients", "local training\non private\ndata"),
         (20.5, "Probe\nmeasurement", "fixed probe\nbank, two\nserver folds"),
         (40.5, "Breadth +\nprofiles", "k-th largest\nsource; excess\nover peers"),
         (60.5, "Bounded\ninfluence", "trust from\nscore; clipping;\n0 ≤ aᵢ ≤ pᵢ"),
         (80.5, "Repair &\nsafety", "revocation;\nhold if ≥ 50%\nflagged")]
for x, head, body in steps:
    key = x in (20.5, 40.5)
    ax.add_patch(FancyBboxPatch((x, 4), 18, 25, boxstyle="round,pad=0.3,rounding_size=1.2", facecolor=SURF,
                                edgecolor=SLOTS[0] if key else INK2, linewidth=1.6 if key else 0.9))
    ax.text(x + 9, 23.5, head, ha="center", va="center", fontsize=9.5, color=INK, weight="bold", linespacing=1.1)
    ax.text(x + 9, 11.5, body, ha="center", va="center", fontsize=8.8, color=INK2, linespacing=1.25)
for x in (18.8, 38.8, 58.8, 78.8):
    ax.add_patch(FancyArrowPatch((x, 16.5), (x + 1.6, 16.5), arrowstyle="-|>", mutation_scale=10, color=INK2, linewidth=1.1))
ax.annotate("", xy=(9.5, 3.0), xytext=(89.5, 3.0), arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=1.0,
                                                                    connectionstyle="arc3,rad=-0.12"))
ax.text(50, -5.5, "new global model broadcast to all clients for the next round", ha="center", fontsize=8.8, color=INK2)
save(fig, "fig0_method_overview")
