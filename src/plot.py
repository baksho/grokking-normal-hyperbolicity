"""Produce the two paper figures from the diagnostic CSVs.

Figure 1: test/train accuracy and sigma_min^+(J) vs step (single seed).
Figure 2: (a) the k smallest singular values; (b) multi-seed robustness.
"""
import argparse, csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

STYLE = {
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "xtick.direction": "in", "ytick.direction": "in",
    "axes.linewidth": 1.0,
}
BAND = (6600, 17400)
RED, BLUE = "#C1352B", "#1F5FA6"


def load_single(path):
    steps, tra, tea, S = [], [], [], None
    with open(path) as f:
        rd = csv.DictReader(f)
        ncol = sum(1 for c in rd.fieldnames if c.startswith("s") and c[1:].isdigit())
        S = [[] for _ in range(ncol)]
        for r in rd:
            steps.append(int(r["step"])); tra.append(float(r["train_acc"])); tea.append(float(r["test_acc"]))
            for i in range(ncol):
                S[i].append(float(r[f"s{i+1}"]))
    return steps, tra, tea, S


def load_multi(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            s = int(r["seed"]); d.setdefault(s, {"step": [], "te": [], "sm": []})
            d[s]["step"].append(int(r["step"])); d[s]["te"].append(float(r["test_acc"]))
            d[s]["sm"].append(float(r["sigma_min"]))
    return d


def fig1(steps, tra, tea, S, out):
    plt.rcParams.update({**STYLE, "font.size": 14, "axes.labelsize": 16,
                         "axes.titlesize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "legend.fontsize": 12})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xmax = max(steps)
    ax.axvspan(*BAND, color="#F2C14E", alpha=0.22, lw=0, label="grokking transition")
    ax.plot(steps, tra, color="#666", lw=1.4, ls=(0, (4, 2)), label="train acc")
    ax.plot(steps, tea, color=BLUE, lw=2.4, label="test acc")
    ax.set_xlabel("training step"); ax.set_ylabel("accuracy")
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True)
    ax2 = ax.twinx()
    ax2.plot(steps, S[0], color=RED, lw=2.4, marker="o", ms=3.2, label=r"$\sigma_{\min}^{+}(J)$")
    ax2.set_ylabel(r"$\sigma_{\min}^{+}(J)$  (normal restoring rate)", color=RED)
    ax2.set_ylim(0, 0.26); ax2.tick_params(axis="y", colors=RED, direction="in")
    ax2.yaxis.set_minor_locator(AutoMinorLocator()); ax2.spines["right"].set_color(RED)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", framealpha=0.95, edgecolor="#ccc")
    ax.set_title(r"No collapse of $\sigma_{\min}^{+}(J)$ at the grokking transition")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")


def fig2(steps, S, multi, out):
    plt.rcParams.update({**STYLE, "font.size": 13, "axes.labelsize": 14,
                         "axes.titlesize": 13.5, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "legend.fontsize": 9.5})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    xmax = max(steps)
    # (a) spectrum
    axA.axvspan(*BAND, color="#F2C14E", alpha=0.22, lw=0)
    cols = ["#1b1b1b", RED, "#2166AC", "#1B7837", "#8552A1", "#E08214"]
    for i in range(min(6, len(S))):
        axA.plot(steps[1:], S[i][1:], color=cols[i], lw=1.5, label=fr"$\sigma_{i+1}^+$")
    axA.set_xlabel("training step"); axA.set_ylabel(r"6 smallest singular values of $J$")
    axA.set_xlim(0, xmax); axA.set_ylim(0, 0.28)
    axA.xaxis.set_minor_locator(AutoMinorLocator()); axA.yaxis.set_minor_locator(AutoMinorLocator())
    axA.tick_params(which="both", direction="in", top=True, right=True)
    axA.legend(loc="lower center", ncol=6, framealpha=0.95, edgecolor="#ccc",
               columnspacing=0.8, handlelength=1.2, handletextpad=0.4, bbox_to_anchor=(0.5, -0.005))
    axA.set_title("(a) low end of the spectrum")
    # (b) multiseed
    seeds = sorted(multi); ms = multi[seeds[0]]["step"]
    TE = np.array([multi[s]["te"] for s in seeds]); SM = np.array([multi[s]["sm"] for s in seeds])
    te_m, te_s, sm_m, sm_s = TE.mean(0), TE.std(0), SM.mean(0), SM.std(0)
    axB.set_xlabel("training step"); axB.set_ylabel("test accuracy (mean $\\pm$ s.d.)")
    axB.set_xlim(0, max(ms)); axB.set_ylim(0, 1.0)
    axB.plot(ms, te_m, color=BLUE, lw=2.2, label="test acc")
    axB.fill_between(ms, np.clip(te_m - te_s, 0, 1), np.clip(te_m + te_s, 0, 1), color=BLUE, alpha=0.18, lw=0)
    axB.xaxis.set_minor_locator(AutoMinorLocator()); axB.yaxis.set_minor_locator(AutoMinorLocator())
    axB.tick_params(which="both", direction="in", top=True)
    axB2 = axB.twinx()
    for s in seeds:
        axB2.plot(ms[1:], multi[s]["sm"][1:], color=RED, lw=0.7, alpha=0.30)
    axB2.plot(ms[1:], sm_m[1:], color=RED, lw=2.4, label=r"$\sigma_{\min}^{+}(J)$")
    axB2.fill_between(ms[1:], (sm_m - sm_s)[1:], (sm_m + sm_s)[1:], color=RED, alpha=0.18, lw=0)
    axB2.set_ylabel(r"$\sigma_{\min}^{+}(J)$", color=RED); axB2.set_ylim(0, 0.28)
    axB2.tick_params(axis="y", colors=RED, direction="in"); axB2.spines["right"].set_color(RED)
    axB2.yaxis.set_minor_locator(AutoMinorLocator())
    h1, l1 = axB.get_legend_handles_labels(); h2, l2 = axB2.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, loc="center right", framealpha=0.95, edgecolor="#ccc")
    axB.set_title("(b) robustness across 5 seeds")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results_seed0.csv")
    ap.add_argument("--multiseed", default="results/multiseed.csv")
    ap.add_argument("--figdir", default="paper/figures")
    args = ap.parse_args()
    steps, tra, tea, S = load_single(args.results)
    fig1(steps, tra, tea, S, f"{args.figdir}/fig1_sigma_min.png")
    multi = load_multi(args.multiseed)
    fig2(steps, S, multi, f"{args.figdir}/fig2_spectrum_multiseed.png")
    print(f"saved figures to {args.figdir}/")


if __name__ == "__main__":
    main()
