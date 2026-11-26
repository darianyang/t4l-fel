"""Export the 2D free-energy surface F(S_path, Z_path) of an ensemble as CSV.

  python export_fes.py [--ens bioemu1-v1.2_100k]   -> analysis/fes2d_<ens>_<bin>.csv

Columns: s_path, z_path (nm^2), F_kcal_mol. Rows are bin centres of a 0.01 x 0.01 grid
(S 0.10-0.90, Z -0.05-0.65 nm^2, covering all samples); only populated bins are written.
F = -kT ln p at 298 K, shifted so the global minimum is 0. Reads analysis/cv_<ens>.npz
(from analyze.py); physicality-rejected samples are excluded.
"""

import argparse

import numpy as np

from plot_fes import KT, OUT


def main(a):

    S_EDGES = np.round(np.arange(0.10, 0.90 + 1e-9, a.bin), 3)
    Z_EDGES = np.round(np.arange(-0.05, 0.65 + 1e-9, a.bin), 3)

    d = np.load(OUT / f"cv_{a.ens}.npz")
    s, z = d["s"][d["ok"]], d["z"][d["ok"]]
    H, _, _ = np.histogram2d(s, z, bins=[S_EDGES, Z_EDGES])
    assert H.sum() == len(s), "samples fall outside the grid"
    with np.errstate(divide="ignore"):
        F = -KT * np.log(H / H.sum())
    F -= F[np.isfinite(F)].min()
    sc, zc = 0.5 * (S_EDGES[1:] + S_EDGES[:-1]), 0.5 * (Z_EDGES[1:] + Z_EDGES[:-1])
    S, Z = np.meshgrid(sc, zc, indexing="ij")
    m = np.isfinite(F)
    out = OUT / f"fes2d_{a.ens}_{a.bin:.2f}.csv"
    np.savetxt(out, np.c_[S[m], Z[m], F[m]], delimiter=",", fmt=["%.3f", "%.3f", "%.4f"],
               header="s_path,z_path,F_kcal_mol", comments="")
    print(f"wrote {out}: {m.sum()} populated bins of {m.size}, n={len(s)}, "
          f"F range 0-{F[m].max():.2f} kcal/mol")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bin", type=float, default=0.03, help="bin width for S and Z")
    p.add_argument("--ens", default="bioemu1-v1.2_100k")
    main(p.parse_args())
