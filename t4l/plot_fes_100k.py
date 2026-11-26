"""Free-energy landscape of the 100k BioEmu1 v1.2 ensemble vs PathMetaD (Wang et al., eLife 2016).

  python analyze.py --reuse          # computes analysis/cv_bioemu1-v1.2_100k.npz if missing
  python plot_fes_100k.py [--ens bioemu1-v1.2_100k] -> analysis/fes_spath_100k.{png,pdf}

(a) F(S_path): 100k ensemble (band = +/- 1 sd over bootstraps), the 10k v1.2 run (dashed) as a
    convergence check, and the digitized PathMetaD profile. Dashed grey line = BioEmu sampling
    limit kT ln(N / MIN_COUNT). All profiles shifted to min = 0.
(b) F(S_path, Z_path) of the 100k ensemble; 3DMV / 2LCB projected for reference.
Uses ../default.mplstyle, T = 298 K.
"""

import argparse
import json

import matplotlib
import numpy as np

from plot_fes import BINS, KT, METAD, MIN_COUNT, MUTED, OUT, S_E, S_I, SEP, STYLE, profile

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = "#1baf7a"  # BioEmu1 v1.2, same slot as in fes_spath_1d.png


def dg_split(s, lo):
    pe = (s > lo).mean()
    return -KT * np.log(pe / (1 - pe)), int((s > lo).sum())


def main(a):
    summ = json.load(open(OUT / "summary.json"))
    ref = summ["references"]
    d = np.load(OUT / f"cv_{a.ens}.npz")
    s, z = d["s"][d["ok"]], d["z"][d["ok"]]
    rng = np.random.default_rng(0)
    plt.style.use(STYLE)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.1, 1]))

    # (a) 1D profile
    ax.axvspan(*S_E, color="0.93", zorder=0)
    ax.text(np.mean(S_E), 9.7, "E basin", ha="center", va="top", fontsize=11, color=MUTED)
    ax.axvline(SEP, color=MUTED, ls=":", lw=1.5)
    ax.axvline(S_I, color=MUTED, ls=(0, (1, 3)), lw=1)
    ax.text(S_I - 0.008, 9.7, "I$_{0.36}$", ha="right", va="top", fontsize=11, color=MUTED)
    sm, Fm = np.loadtxt(METAD).T
    Fm = np.array([np.median(Fm[max(0, i - 4):i + 5]) for i in range(len(Fm))])
    ax.plot(sm, Fm, color="k", lw=2.5, zorder=9, label="PathMetaD (ΔG 3.5)")
    small = OUT / "cv_bioemu1-v1.2.npz"
    if a.ens != "bioemu1-v1.2" and small.is_file():
        d10 = np.load(small)
        x, F, _ = profile(d10["s"][d10["ok"]], rng)
        ax.plot(x, F, "--", color=COL, lw=1.5, zorder=5,
                label=f"v1.2, n={d10['ok'].sum() / 1000:.0f}k (ΔG {dg_split(d10['s'], SEP)[0]:.1f})")
    x, F, sd = profile(s, rng)
    g46, _ = dg_split(s, SEP)
    ax.fill_between(x, F - sd, F + sd, color=COL, alpha=0.25, lw=0, zorder=6)
    ax.plot(x, F, "-", color=COL, lw=2.5, zorder=7,
            label=f"v1.2, n={len(s) / 1000:.0f}k (ΔG {g46:.1f})")
    flim = KT * np.log(len(s) / MIN_COUNT)
    ax.axhline(flim, color=MUTED, ls="--", lw=1)
    ax.text(0.60, flim + 0.12, f"sampling limit (n={len(s) / 1000:.0f}k)", fontsize=10,
            color=MUTED, va="bottom")
    for k, m in (("3DMV", "v"), ("2LCB", "^")):
        ax.plot(ref[k]["s"], np.full(len(ref[k]["s"]), -0.5), m, color="0.35", ms=7,
                ls="none", clip_on=False, zorder=10)
    ax.text(0.225, -0.5, "3DMV", va="center", fontsize=10, color="0.35")
    ax.text(0.725, -0.5, "2LCB", ha="right", va="center", fontsize=10, color="0.35")
    ax.set(xlim=(0.15, 0.85), ylim=(-0.9, 10))
    ax.set_xlabel("S$_\\mathrm{path}$")
    ax.set_ylabel("F (kcal/mol)")
    ax.legend(fontsize=10, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False,
              handlelength=1.5, columnspacing=1.0)
    ax.text(-0.13, 1.02, "a", transform=ax.transAxes, fontsize=16, fontweight="bold")

    # (b) 2D F(S, Z)
    H, xe, ye = np.histogram2d(s, z, bins=[np.linspace(0.15, 0.85, 71), np.linspace(-0.05, 0.3, 50)])
    with np.errstate(divide="ignore"):
        F2 = -KT * np.log(H / H.sum())
    F2 -= F2[np.isfinite(F2)].min()
    im = bx.pcolormesh(xe, ye, np.ma.masked_invalid(F2.T), cmap="Blues_r", vmin=0, vmax=flim,
                       rasterized=True)
    plt.colorbar(im, ax=bx, label="F (kcal/mol)")
    bx.axvline(SEP, color=MUTED, ls=":", lw=1.5)
    for k, m in (("3DMV", "v"), ("2LCB", "^")):
        bx.plot(ref[k]["s"], ref[k]["z"], m, color="#eb6834", mec="k", mew=0.8, ms=8, ls="none",
                label=f"{k} ({'G' if k == '3DMV' else 'E'})")
    bx.set(xlim=(0.15, 0.85), ylim=(-0.05, 0.3))
    bx.set_xlabel("S$_\\mathrm{path}$")
    bx.set_ylabel("Z$_\\mathrm{path}$ (nm$^2$)")
    bx.legend(fontsize=10, loc="upper right", frameon=False)
    bx.text(-0.15, 1.02, "b", transform=bx.transAxes, fontsize=16, fontweight="bold")

    fig.tight_layout()
    stem = "fes_spath_100k" if a.ens.endswith("100k") else f"fes_spath_{a.ens}"
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    g55, n55 = dg_split(s, 0.55)
    print(f"{a.ens}: n={len(s)}  dG(S>0.46)={g46:.2f}  dG(S>0.55)={g55:.2f} (n_E={n55})  "
          f"max S={s.max():.3f}  wrote {OUT / stem}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ens", default="bioemu1-v1.2_100k")
    main(p.parse_args())
