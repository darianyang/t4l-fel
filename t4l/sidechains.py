"""HPacker side-chain reconstruction for BioEmu1 backbone samples (fixed backbone, no MD relax).

  python sidechains.py prep     runs/bioemu1/v1.2     # backbone-only PDB per frame
  python sidechains.py run      runs/bioemu1/v1.2 --nproc 32 [--threads 4] [--gpu ""]   # HPacker workers
  python sidechains.py assemble runs/bioemu1/v1.2     # -> hpacker/samples_sidechain_rec.{xtc,pdb}

Mirrors bioemu.sidechain_relax.reconstruct_sidechains (backbone atoms only, CB discarded) but
runs HPacker in parallel shards that load the networks once (hpacker_batch.py).
"""

import argparse
import os
import subprocess
from pathlib import Path

import mdtraj
import numpy as np

HERE = Path(__file__).resolve().parent
HPACKER_PY = Path.home() / ".hpacker_venv" / "bin" / "python"


def prep(run):
    t = mdtraj.load(run / "samples.xtc", top=run / "topology.pdb")
    bb = t.atom_slice(t.topology.select("backbone"))
    d = run / "hpacker" / "bb"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(bb.n_frames):
        f = d / f"frame_{i:05d}_bb.pdb"
        if not f.is_file():
            bb[i].save_pdb(str(f))
    print(f"prep: {bb.n_frames} backbone frames in {d}")


def run_workers(run, nproc, gpu, threads=4):
    d = run / "hpacker"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads))
    procs = []
    for i in range(nproc):
        log = open(d / f"shard_{i:03d}.log", "a")
        procs.append(subprocess.Popen(
            [str(HPACKER_PY), str(HERE / "hpacker_batch.py"), str(d / "bb"), str(d / "heavy"),
             "--shard", str(i), "--nshards", str(nproc)], stdout=log, stderr=subprocess.STDOUT,
            env=env))
    codes = [p.wait() for p in procs]
    print("worker exit codes:", codes)


def assemble(run):
    d = run / "hpacker"
    files = sorted((d / "heavy").glob("frame_*_heavy.pdb"))
    ref = mdtraj.load_pdb(str(files[0]))
    xyz, idx, skipped = [], [], []
    for f in files:
        t = mdtraj.load_pdb(str(f))
        if t.topology != ref.topology:
            skipped.append(f.name)
            continue
        xyz.append(t.xyz[0])
        idx.append(int(f.name.split("_")[1]))
    # bioemu1 topology.pdb numbers residues from 0; shift to T4L numbering (Met1) used by path.pdb
    shift = 1 - min(r.resSeq for r in ref.topology.residues)
    for r in ref.topology.residues:
        r.resSeq += shift
    traj = mdtraj.Trajectory(np.array(xyz), ref.topology)
    traj.save_xtc(str(d / "samples_sidechain_rec.xtc"))
    ref.save_pdb(str(d / "samples_sidechain_rec.pdb"))
    np.save(d / "frame_index.npy", np.array(idx))
    print(f"assemble: {traj.n_frames} frames ({len(skipped)} skipped: topology mismatch)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prep", "run", "assemble"])
    p.add_argument("run", type=Path)
    p.add_argument("--nproc", type=int, default=32)
    p.add_argument("--gpu", default="")
    p.add_argument("--threads", type=int, default=4, help="OMP/MKL threads per worker")
    a = p.parse_args()
    if a.cmd == "prep":
        prep(a.run)
    elif a.cmd == "run":
        run_workers(a.run, a.nproc, a.gpu, a.threads)
    else:
        assemble(a.run)
