"""Project BioEmu ensembles of T4L L99A onto the eLife 2016 path CVs and compare with the paper.

  python analyze.py                      # all ensembles found under runs/
  python analyze.py --only bioemu2

For each ensemble: S_path (s/N), Z_path (nm^2), F114 chi1, physicality flag; saved to
analysis/cv_<name>.npz. Free energies use kT at 298 K; dG(E-G) splits at the paper's
separatrix S_path = 0.46 (bootstrap over samples for the error).
"""

import argparse
import json
from pathlib import Path

import matplotlib
import mdtraj
import numpy as np

from spath import path_cvs

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS, OUT = HERE / "runs", HERE / "analysis"
KT = 0.0019872 * 298.0  # kcal/mol
SEP = 0.46  # eLife separatrix between G and E basins on S_path
# Reference values from Wang et al. eLife 2016 (Table 1, Figs 2-3) and Bouvignies et al. 2011
PAPER = dict(dG_nmr=2.1, dG_pathmetad=3.5, dG_inmetad=2.9, s_G=0.2, s_E=(0.55, 0.83), s_I=0.36)


def f114_chi1(t):
    top = t.topology
    ids = [top.select(f"resSeq 114 and name {n}")[0] for n in ("N", "CA", "CB", "CG")]
    return np.degrees(mdtraj.compute_dihedrals(t, [ids])[:, 0])


def load_ensembles(only=None):
    ens = {}
    b2 = sorted((RUNS / "bioemu2").glob("seed*/chunk_*/samples.xtc"))
    if b2:
        ens["bioemu2"] = b2
    for d in sorted((RUNS / "bioemu1").glob("v1.*")):
        # single run (v1.2/hpacker/...) or a run split over GPUs (v1.2_100k/part*/hpacker/...)
        fs = sorted(d.glob("hpacker/samples_sidechain_rec.xtc")) or \
            sorted(d.glob("part*/hpacker/samples_sidechain_rec.xtc"))
        if fs:
            ens[f"bioemu1-{d.name}"] = fs
    return {k: v for k, v in ens.items() if only is None or k.startswith(only)}


def compute(name, files, reuse=False):
    if reuse and (OUT / f"cv_{name}.npz").is_file():
        return dict(np.load(OUT / f"cv_{name}.npz"))
    s, z, chi, ok = [], [], [], []
    for f in files:
        top = f.parent / ("topology.cif" if (f.parent / "topology.cif").is_file()
                          else "samples_sidechain_rec.pdb")
        t = mdtraj.load(f, top=top)
        t = t.atom_slice(t.topology.select("protein"))
        cv = path_cvs(t)
        s.append(cv["s_norm"]), z.append(cv["z"]), chi.append(f114_chi1(t))
        pj = f.parent / "physicality.json"
        if pj.is_file():
            ok.append([r["status"] != "rejected" for r in json.load(open(pj))])
        else:
            ok.append(np.ones(t.n_frames, bool))
    d = dict(s=np.concatenate(s), z=np.concatenate(z), chi1=np.concatenate(chi),
             ok=np.concatenate(ok))
    np.savez(OUT / f"cv_{name}.npz", **d)
    return d


def references():
    ref = {}
    for name in ("3DMV", "2LCB"):
        t = mdtraj.load(HERE / "inputs" / "pdb" / f"{name}.pdb")
        t = t.atom_slice(t.topology.select("protein"))
        cv = path_cvs(t)
        ref[name] = dict(s=cv["s_norm"], z=cv["z"], chi1=f114_chi1(t))
    return ref


def dG_split(s, rng, nboot=500):
    def dg(x):
        pe = (x > SEP).mean()
        return np.inf if pe == 0 else -KT * np.log(pe / (1 - pe))
    boots = [dg(rng.choice(s, len(s))) for _ in range(nboot)]
    boots = np.array([b for b in boots if np.isfinite(b)])
    return dg(s), (boots.std() if len(boots) else np.nan), int((s > SEP).sum())


def fes_1d(s, bins):
    h, e = np.histogram(s, bins=bins)
    with np.errstate(divide="ignore"):
        f = -KT * np.log(h / h.sum())
    return 0.5 * (e[1:] + e[:-1]), f - np.nanmin(f[np.isfinite(f)])


def main(a):
    OUT.mkdir(exist_ok=True)
    ens = load_ensembles(a.only)
    ref = references()
    rng = np.random.default_rng(0)
    data, summary = {}, {"paper": PAPER, "references": {
        k: {kk: np.round(vv, 3).tolist() for kk, vv in v.items()} for k, v in ref.items()}}
    for name, files in ens.items():
        d = compute(name, files, a.reuse)
        data[name] = d
        s = d["s"][d["ok"]]
        g, ge, nE = dG_split(s, rng)
        chi_E = d["chi1"][d["ok"]][s > SEP]
        summary[name] = dict(
            n_total=int(len(d["s"])), n_physical=int(d["ok"].sum()), n_E=nE,
            p_E=float(nE / len(s)), dG_E_minus_G=float(g), dG_err=float(ge),
            s_quantiles=np.round(np.quantile(s, [0.01, 0.25, 0.5, 0.75, 0.99]), 3).tolist(),
            frac_F114_trans_in_E=float(((chi_E > 120) | (chi_E < -120)).mean()) if len(chi_E) else None,
            frac_F114_trans_all=float(((d["chi1"] > 120) | (d["chi1"] < -120)).mean()),
        )
        print(name, json.dumps(summary[name]))
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    if data:
        plot(data, ref)


def plot(data, ref):
    bins = np.linspace(0.1, 0.95, 52)
    cols = {"bioemu2": "#1f77b4", "bioemu1-v1.2": "#d62728", "bioemu1-v1.1": "#ff7f0e"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    # (a) 1D FES along S_path
    for name, d in data.items():
        x, f = fes_1d(d["s"][d["ok"]], bins)
        ax[0].plot(x, f, "-o", ms=3, color=cols.get(name), label=f"{name} (n={d['ok'].sum()})")
    ax[0].axvline(SEP, color="k", ls=":", lw=1)
    ax[0].axvspan(*PAPER["s_E"], color="0.85", zorder=0, label="eLife E basin (0.55-0.83)")
    ax[0].axhline(PAPER["dG_pathmetad"], color="0.4", ls="--", lw=1, label="eLife ΔG PathMetaD 3.5")
    ax[0].axhline(PAPER["dG_nmr"], color="g", ls="--", lw=1, label="NMR ΔG 2.1")
    for k, m, c in (("3DMV", "v", "red"), ("2LCB", "^", "blue")):
        ax[0].plot(ref[k]["s"], np.full(len(ref[k]["s"]), -0.4), m, color=c, ms=6,
                   label=f"{k} ({'G' if k == '3DMV' else 'E'})")
    ax[0].set(xlabel="S_path", ylabel="F (kcal/mol, 298 K)", xlim=(0.1, 0.95), ylim=(-1, 8))
    ax[0].legend(fontsize=7, loc="upper center")
    # (b) 2D F(S, Z) for BioEmu1 v1.2 (the only ensemble that reaches E), else the first one
    name = "bioemu1-v1.2" if "bioemu1-v1.2" in data else next(iter(data))
    d = data[name]
    H, xe, ye = np.histogram2d(d["s"][d["ok"]], d["z"][d["ok"]], bins=[bins, np.linspace(-0.05, 0.3, 50)])
    with np.errstate(divide="ignore"):
        F = -KT * np.log(H / H.sum())
    F -= F[np.isfinite(F)].min()
    im = ax[1].pcolormesh(xe, ye, F.T, cmap="viridis", vmax=6)
    plt.colorbar(im, ax=ax[1], label="F (kcal/mol)")
    for k, m, c in (("3DMV", "v", "red"), ("2LCB", "^", "cyan")):
        ax[1].plot(ref[k]["s"], ref[k]["z"], m, color=c, mec="k", ms=7, ls="none", label=k)
    ax[1].axvline(SEP, color="w", ls=":", lw=1)
    ax[1].set(xlabel="S_path", ylabel="Z_path (nm²)", title=f"{name}: F(S, Z)")
    ax[1].legend(fontsize=7)
    # (c) F114 chi1 vs S_path
    for name, d in data.items():
        ax[2].plot(d["s"], d["chi1"], ".", ms=2, alpha=0.4, color=cols.get(name))
        ax[2].plot([], [], "o", ms=4, color=cols.get(name), label=name)
    for k, m, c in (("3DMV", "v", "red"), ("2LCB", "^", "blue")):
        ax[2].plot(ref[k]["s"], ref[k]["chi1"], m, color=c, mec="k", ms=7, ls="none", label=k)
    ax[2].axvline(SEP, color="k", ls=":", lw=1)
    ax[2].set(xlabel="S_path", ylabel="F114 χ1 (deg)", ylim=(-180, 180), xlim=(0.1, 0.95))
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fes_spath.png", dpi=150)
    print("wrote", OUT / "fes_spath.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None)
    p.add_argument("--reuse", action="store_true", help="load cached analysis/cv_<name>.npz")
    main(p.parse_args())
