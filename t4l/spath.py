"""Path collective variables (S_path, Z_path) of Wang, Papaleo & Lindorff-Larsen, eLife 2016.

Re-implements PLUMED `PATHMSD REFERENCE=path.pdb LAMBDA=56.0` in numpy, using the authors'
deposited 31-frame MOIL reference path (github.com/KULL-Centre/papers, 2016/T4L-eLIFE-Wang-et-al):

  * each frame has 218 atoms; occupancy = alignment weight (CA of residues 94-125),
    beta = displacement weight (heavy atoms of residues 98-125);
  * MSD_i (nm^2): centre with alignment weights, Kabsch-rotate on alignment atoms,
    weighted mean square deviation over displacement atoms (PLUMED RMSD TYPE=OPTIMAL, squared);
  * s = sum_i i exp(-lambda MSD_i) / sum_i exp(-lambda MSD_i)   (i = 1..31)
    z = -1/lambda ln sum_i exp(-lambda MSD_i)                    (nm^2)

The paper plots S_path on a 0-1 scale; `s_norm` = s / N. With this, 3DMV (G_Xray) gives 0.21 and
the 2LCB models (E_ROSETTA) 0.74-0.83, matching the G/E basins at ~0.2/~0.75 in eLife Fig. 2.
((s - 1) / (N - 1) would shift values by <= 0.03.)

  python spath.py traj.xtc --top topology.cif [--out cv.npz]
"""

import argparse
from pathlib import Path

import mdtraj
import numpy as np

PATH_PDB = Path(__file__).resolve().parent / "inputs" / "elife_path" / "path.pdb"
LAMBDA = 56.0  # nm^-2
CHARMM_TO_PDB = {("ILE", "CD"): "CD1"}


def load_path(path_pdb=PATH_PDB):
    """Return (keys, xyz [N, n_atoms, 3] in nm, w_align, w_disp) for the reference path."""
    frames, cur = [], []
    for line in open(path_pdb):
        if line.startswith("ATOM"):
            cur.append(line)
        elif line.startswith("END") and cur:
            frames.append(cur)
            cur = []
    if cur:
        frames.append(cur)
    keys = []
    for l in frames[0]:
        res, name = l[17:20].strip(), l[12:16].strip()
        keys.append((int(l[22:26]), CHARMM_TO_PDB.get((res, name), name)))
    xyz = np.array([[[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in f]
                    for f in frames]) / 10.0
    w_al = np.array([float(l[54:60]) for l in frames[0]])
    w_di = np.array([float(l[60:66]) for l in frames[0]])
    return keys, xyz, w_al / w_al.sum(), w_di / w_di.sum()


def atom_indices(topology, keys, chain=0):
    lookup = {(a.residue.resSeq, a.name): a.index
              for a in topology.atoms if a.residue.chain.index == chain}
    missing = [k for k in keys if k not in lookup]
    if missing:
        raise KeyError(f"atoms missing from topology: {missing[:10]} ({len(missing)} total)")
    return np.array([lookup[k] for k in keys])


def msd_to_refs(x, refs, w_al, w_di):
    """x [T, n, 3], refs [N, n, 3] -> MSD [T, N] (PLUMED OPTIMAL, squared)."""
    xc = x - np.einsum("n,tnd->td", w_al, x)[:, None]
    rc = refs - np.einsum("n,knd->kd", w_al, refs)[:, None]
    # correlation matrix C[t,k] = sum_n w_al x_n r_n^T ; find R with R x ~ r
    C = np.einsum("n,tna,knb->tkab", w_al, xc, rc)
    U, _, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.einsum("tkab,tkbc->tkac", U, Vt)))
    D = np.ones(U.shape[:-1])
    D[..., -1] = d
    R = np.einsum("tkab,tkb,tkbc->tkca", U, D, Vt)  # rotation applied as R @ x
    xr = np.einsum("tkab,tnb->tkna", R, xc)
    return np.einsum("n,tknd->tk", w_di, (xr - rc[None]) ** 2)


def path_cvs(traj, lam=LAMBDA, chunk=2000):
    keys, refs, w_al, w_di = load_path()
    idx = atom_indices(traj.topology, keys)
    N = len(refs)
    s, z = [], []
    for i in range(0, traj.n_frames, chunk):
        m = msd_to_refs(traj.xyz[i:i + chunk, idx].astype(np.float64), refs, w_al, w_di)
        a = -lam * m
        amax = a.max(1, keepdims=True)
        w = np.exp(a - amax)
        s.append((w * np.arange(1, N + 1)).sum(1) / w.sum(1))
        z.append(-(np.log(w.sum(1)) + amax[:, 0]) / lam)
    s, z = np.concatenate(s), np.concatenate(z)
    return dict(s=s, s_norm=s / N, z=z)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("traj")
    p.add_argument("--top")
    p.add_argument("--out")
    a = p.parse_args()
    t = mdtraj.load(a.traj, top=a.top) if a.top else mdtraj.load(a.traj)
    cv = path_cvs(t)
    for k, v in cv.items():
        print(k, np.round(v[:20], 3))
    if a.out:
        np.savez(a.out, **cv)
