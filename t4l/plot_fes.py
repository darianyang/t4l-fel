"""F(S_path) of the BioEmu ensembles vs the PathMetaD profile of Wang et al., eLife 2016 (Fig. 2).

  python plot_fes.py        # reads analysis/cv_*.npz (from analyze.py) -> analysis/fes_spath_1d.{png,pdf}

Uses ../default.mplstyle. T = 298 K. All profiles are shifted to min = 0 (the G basin).
dG(E-G) in the legend: BioEmu from populations split at S_path = 0.46 (bootstrap sd);
PathMetaD is the paper's value (3.5; integrating the digitized 1D profile gives 3.0).
The literature curve is digitized from the figure (inputs/elife_fes/), not raw data.

Bands are +/- 1 sd over bootstrap resamples. Bins with < MIN_COUNT samples are dropped, so
BioEmu profiles cannot resolve F above ~kT ln(N / MIN_COUNT) (~4.8 kcal/mol for 10k samples),
shown as the dashed 'sampling limit' line.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "analysis"
METAD = HERE / "inputs" / "elife_fes" / "wang2016_pathmetad_fes_digitized.dat"
STYLE = HERE.parent / "default.mplstyle"
KT = 0.0019872 * 298.0
SEP, S_E, S_I = 0.46, (0.55, 0.83), 0.36
BINS = np.linspace(0.15, 0.85, 36)
MIN_COUNT, NBOOT = 3, 300
# categorical slots 1-3 of the dataviz reference palette, in fixed order
ENS = [("bioemu2", "BioEmu2", "#2a78d6"),
       ("bioemu1-v1.1", "BioEmu1 v1.1", "#eb6834"),
       ("bioemu1-v1.2", "BioEmu1 v1.2", "#1baf7a")]
MUTED = "#6b6a66"


def profile(s, rng):
    x = 0.5 * (BINS[1:] + BINS[:-1])
    h = np.histogram(s, BINS)[0]
    hb = np.array([np.histogram(rng.choice(s, len(s)), BINS)[0] for _ in range(NBOOT)])
    with np.errstate(divide="ignore"):
        f = -KT * np.log(h / h.sum())
        fb = -KT * np.log(hb / hb.sum(1, keepdims=True))
    f0 = f[np.isfinite(f)].min()
    f, fb = f - f0, fb - f0
    fb[~np.isfinite(fb)] = np.nan
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(fb, 0)
    keep = h >= MIN_COUNT
    return x[keep], f[keep], sd[keep]


def main():
    summ = json.load(open(OUT / "summary.json"))
    ref = summ["references"]
    rng = np.random.default_rng(0)
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.axvspan(*S_E, color="0.93", zorder=0)
    ax.text(np.mean(S_E), 9.7, "E basin", ha="center", va="top", fontsize=11, color=MUTED)
    ax.axvline(SEP, color=MUTED, ls=":", lw=1.5)
    ax.axvline(S_I, color=MUTED, ls=(0, (1, 3)), lw=1)
    ax.text(S_I - 0.008, 9.7, "I$_{0.36}$", ha="right", va="top", fontsize=11, color=MUTED)

    sm, Fm = np.loadtxt(METAD).T
    # running median over 9 px (~0.01 in S) removes stripe artefacts of the digitized fill
    Fm = np.array([np.median(Fm[max(0, i - 4):i + 5]) for i in range(len(Fm))])
    ax.plot(sm, Fm, color="k", lw=2.5, zorder=9, label="PathMetaD, eLife 2016 (ΔG 3.5)")

    nmax = 0
    for z, (key, label, col) in enumerate(ENS[::-1]):
        f = OUT / f"cv_{key}.npz"
        if not f.is_file():
            continue
        d = np.load(f)
        s = d["s"][d["ok"]]
        nmax = max(nmax, len(s))
        x, F, sd = profile(s, rng)
        dg = summ[key]["dG_E_minus_G"]
        dgs = f"ΔG {dg:.1f}" if np.isfinite(dg) else "ΔG > 4.8"
        ax.fill_between(x, F - sd, F + sd, color=col, alpha=0.2, lw=0, zorder=2 + z)
        ax.plot(x, F, "-", color=col, lw=2, zorder=5 + z, label=f"{label} ({dgs})")

    flim = KT * np.log(nmax / MIN_COUNT)
    ax.axhline(flim, color=MUTED, ls="--", lw=1)
    ax.text(0.28, flim + 0.12, "BioEmu sampling limit", fontsize=10, color=MUTED, va="bottom")
    ax.axhline(2.1, xmin=0.94, xmax=1.0, color="#008300", lw=3)
    ax.text(0.845, 2.1 + 0.15, "NMR ΔG", ha="right", va="bottom", fontsize=10, color="#008300")
    for k, m in (("3DMV", "v"), ("2LCB", "^")):
        ax.plot(ref[k]["s"], np.full(len(ref[k]["s"]), -0.5), m, color="0.35", ms=7,
                ls="none", clip_on=False, zorder=10)
    ax.text(0.225, -0.5, "3DMV", va="center", fontsize=10, color="0.35")
    ax.text(0.725, -0.5, "2LCB", ha="right", va="center", fontsize=10, color="0.35")

    ax.set(xlim=(0.15, 0.85), ylim=(-0.9, 10))
    ax.set_xlabel("S$_\\mathrm{path}$")
    ax.set_ylabel("F (kcal/mol)")
    h, lab = ax.get_legend_handles_labels()
    order = [3, 2, 1, 0]  # BioEmu in slot order, then the literature curve
    ax.legend([h[i] for i in order if i < len(h)], [lab[i] for i in order if i < len(h)],
              fontsize=10, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              frameon=False, handlelength=1.5, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fes_spath_1d.png", bbox_inches="tight")
    fig.savefig(OUT / "fes_spath_1d.pdf", bbox_inches="tight")
    print("wrote", OUT / "fes_spath_1d.png")


if __name__ == "__main__":
    main()
