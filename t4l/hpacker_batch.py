"""Run HPacker over many backbone-only PDBs, loading the networks once per process.

Runs in the HPacker venv (~/.hpacker_venv, created by bioemu's setup_sidechain_relax.sh):
  ~/.hpacker_venv/bin/python hpacker_batch.py <in_dir> <out_dir> --shard i --nshards n

Same settings as bioemu.run_hpacker (reconstruct_sidechains(num_refinement_iterations=5)).
Frames already present in <out_dir> are skipped (checked per frame), so extra workers can be
added to a running job with --reverse; at worst a few frames are packed twice.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from hpacker import HPacker

_cache = {}
_orig_init_models = HPacker._init_models


def _cached_init_models(self, model_dirs):
    if "obj" not in _cache:
        _orig_init_models(self, model_dirs)
        _cache["obj"] = self
        return
    src = _cache["obj"]
    for k in ("model_dirs", "hparams", "data_irreps", "ls_indices", "model", "loss_fn", "device"):
        setattr(self, k, getattr(src, k))


HPacker._init_models = _cached_init_models


def main(a):
    ins = sorted(Path(a.in_dir).glob("frame_*_bb.pdb"), reverse=a.reverse)[a.shard::a.nshards]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    for f in ins:
        o = out / f.name.replace("_bb.pdb", "_heavy.pdb")
        if o.is_file():
            continue
        h = HPacker(str(f))
        h.reconstruct_sidechains(num_refinement_iterations=5)
        tmp = o.with_suffix(".tmp.pdb")
        h.write_pdb(str(tmp))
        os.replace(tmp, o)
        done += 1
        if done % 50 == 0:
            print(f"shard {a.shard}: {done} frames, {(time.time() - t0) / done:.2f} s/frame",
                  flush=True)
    print(f"shard {a.shard}: finished {done} frames in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("in_dir")
    p.add_argument("out_dir")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--reverse", action="store_true",
                   help="walk frames from the end (extra workers alongside a running forward set)")
    sys.exit(main(p.parse_args()))
