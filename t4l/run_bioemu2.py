"""BioEmu2 equilibrium ensemble of apo T4L L99A (3DMV construct) for a first look at the G/E FEL.

  CUDA_VISIBLE_DEVICES=0 python run_bioemu2.py --n 5000 --seed 0    --tag seed0
  CUDA_VISIBLE_DEVICES=1 python run_bioemu2.py --n 5000 --seed 5000 --tag seed5000

Samples are written in chunks (runs/bioemu2/<tag>/chunk_XXXX/{samples.xtc,topology.cif,
physicality.json}) so a run can be analysed while it is going and resumed after a crash
(finished chunks are skipped; the running chunk resumes from its partial batches).
Chunk k uses seed = --seed + k * 1000 (batch i inside a chunk uses chunk_seed + i).
"""

import argparse
import logging
import shutil
import time
from pathlib import Path

from bioemu2 import System, get_msa, load_model, save_frames
from bioemu2.sampling import check_physicality
from bioemu2.sampling import sample_with_embeddings, sampler_from_preset

HERE = Path(__file__).resolve().parent
SYSTEM_YAML = HERE / "inputs" / "t4l_l99a.yaml"
MSA_A3M = HERE / "inputs" / "t4l_l99a.a3m"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("t4l")


def main(args):
    out = HERE / "runs" / "bioemu2" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    system = System.from_yaml(SYSTEM_YAML)
    msa = get_msa(system, source=MSA_A3M)
    model = load_model()
    emb = model.embed(system, msa=msa, mode="equilibrium")
    sampler = sampler_from_preset("edm", mode="equilibrium", num_steps=args.num_steps)

    n_chunks = -(-args.n // args.chunk)
    for k in range(n_chunks):
        cdir = out / f"chunk_{k:04d}"
        if (cdir / "samples.xtc").is_file():
            continue
        n = min(args.chunk, args.n - k * args.chunk)
        t0 = time.time()
        traj = sample_with_embeddings(system, emb, model=model, num_samples=n,
                                      batch_size=args.batch, sampler=sampler,
                                      seed=args.seed + k * 1000, progress=False,
                                      partial_dir=out / "partial" / f"chunk_{k:04d}")
        save_frames(traj, cdir)
        report = check_physicality(traj, system=system)
        report.save(cdir / "physicality.json")
        shutil.rmtree(out / "partial" / f"chunk_{k:04d}", ignore_errors=True)
        dt = time.time() - t0
        log.info("chunk %d/%d: %d samples in %.0f s (%.2f s/sample), physicality pass %d/%d",
                 k + 1, n_chunks, n, dt, dt / n, int(report.passed.sum()), n)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-steps", type=int, default=None, help="EDM steps (preset default if unset)")
    p.add_argument("--tag", default="seed0")
    main(p.parse_args())
