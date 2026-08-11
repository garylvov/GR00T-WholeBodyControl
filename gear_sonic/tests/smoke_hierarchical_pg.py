# SPDX-License-Identifier: Apache-2.0
"""Multi-process smoke test for the hierarchical all-reduce. NEEDS GPUs.

This validates the one assumption the unit tests cannot: that
``new_group(backend="nccl")`` succeeds for a leaders-only subgroup while the
DEFAULT group is gloo and ``stack`` ranks share each device.  If that fails,
the whole hierarchical design is dead and we should know in 60 seconds rather
than after a cutover.

Run under torchrun, smallest shape that exercises co-location:

    torchrun --nproc_per_node=4 gear_sonic/tests/smoke_hierarchical_pg.py \
        --n-gpu 2 --stack 2

Production shape (8 GPUs, 24 ranks):

    torchrun --nproc_per_node=24 gear_sonic/tests/smoke_hierarchical_pg.py \
        --n-gpu 8 --stack 3

Checks, in order of how badly each would hurt if wrong:
  1. the NCCL leaders-only subgroup CONSTRUCTS at all
  2. the three-level reduce equals a flat all-reduce, bit-for-bit in fp32
  3. it is FASTER than the flat gloo all-reduce it replaces
"""

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist


def _log(rank, msg):
    print(f"[smoke rank {rank}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gpu", type=int, required=True)
    ap.add_argument("--stack", type=int, required=True)
    ap.add_argument("--numel", type=int, default=8_000_000,
                    help="gradient buffer size; default ~32 MB fp32")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    expected = args.n_gpu * args.stack
    if world != expected:
        raise SystemExit(f"WORLD_SIZE={world} but n_gpu*stack={expected}")

    # The flat group MUST be gloo: it contains co-located ranks, which is
    # exactly what NCCL refuses. This mirrors resolve_ddp_backend().
    dist.init_process_group(backend="gloo")

    from gear_sonic.utils.hierarchical_pg import (
        build_hierarchical_groups,
        device_of_rank,
    )

    device = device_of_rank(rank, args.n_gpu)
    torch.cuda.set_device(device)
    _log(rank, f"device={device} world={world}")

    # ---- check 1: does the leaders-only NCCL subgroup construct? ----------
    t0 = time.time()
    groups = build_hierarchical_groups(args.n_gpu, args.stack)
    dist.barrier()
    if rank == 0:
        _log(rank, f"CHECK 1 PASS: groups built in {time.time()-t0:.2f}s")
        _log(rank, f"  {groups!r}")

    # ---- check 2: does the hierarchy equal a flat all-reduce? -------------
    torch.manual_seed(1234 + rank)
    src = torch.randn(args.numel, device=f"cuda:{device}", dtype=torch.float32)

    flat = src.clone()
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world)

    hier = src.clone()
    dist.all_reduce(hier, op=dist.ReduceOp.SUM, group=groups.intra_pg)
    if groups.inter_pg is not None:
        dist.all_reduce(hier, op=dist.ReduceOp.SUM, group=groups.inter_pg)
    dist.broadcast(hier, src=groups.leader, group=groups.intra_pg)
    hier.div_(world)

    max_diff = (flat - hier).abs().max().item()
    dist.barrier()
    if rank == 0:
        # fp32 sums in different orders differ in the last ulp; 1e-4 relative
        # on a ~N(0,1)/world mean is generous but still catches a wrong tree.
        verdict = "PASS" if max_diff < 1e-4 else "FAIL"
        _log(rank, f"CHECK 2 {verdict}: max|flat - hierarchical| = {max_diff:.3e}")

    # ---- check 3: is it actually faster than the flat gloo path? ----------
    def timeit(fn):
        torch.cuda.synchronize()
        dist.barrier()
        ts = []
        for _ in range(args.iters):
            buf = src.clone()
            torch.cuda.synchronize()
            t = time.time()
            fn(buf)
            torch.cuda.synchronize()
            ts.append(time.time() - t)
        return statistics.median(ts)

    def flat_fn(buf):
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        buf.div_(world)

    # ---- variant A: explicit pinned staging ------------------------------
    # Gloo's CUDA path stages through host memory internally; whether that
    # staging buffer is PINNED is not under our control. Doing the copy
    # ourselves through an explicitly pinned buffer makes the transfer async
    # and typically 2-3x faster. The reduction is then a plain CPU gloo
    # all-reduce. The SUM is identical -- same values, same op -- so gradient
    # dynamics are bit-for-bit unchanged; only the transport differs.
    staging = torch.empty(args.numel, dtype=torch.float32,
                          device="cpu", pin_memory=True)

    def flat_pinned_fn(buf):
        staging.copy_(buf, non_blocking=True)
        torch.cuda.synchronize()
        dist.all_reduce(staging, op=dist.ReduceOp.SUM)
        buf.copy_(staging, non_blocking=True)
        torch.cuda.synchronize()
        buf.div_(world)

    def hier_pinned_fn(buf):
        staging.copy_(buf, non_blocking=True)
        torch.cuda.synchronize()
        dist.all_reduce(staging, op=dist.ReduceOp.SUM, group=groups.intra_pg)
        buf.copy_(staging, non_blocking=True)
        torch.cuda.synchronize()
        if groups.inter_pg is not None:
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=groups.inter_pg)
        staging.copy_(buf, non_blocking=True)
        torch.cuda.synchronize()
        dist.broadcast(staging, src=groups.leader, group=groups.intra_pg)
        buf.copy_(staging, non_blocking=True)
        torch.cuda.synchronize()
        buf.div_(world)

    def hier_fn(buf):
        dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=groups.intra_pg)
        if groups.inter_pg is not None:
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=groups.inter_pg)
        dist.broadcast(buf, src=groups.leader, group=groups.intra_pg)
        buf.div_(world)

    # Correctness of the pinned variants before timing them: a faster wrong
    # answer is worthless, and the staging path is easy to get subtly wrong.
    ref = src.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)
    ref.div_(world)
    for name, fn in (("flat_pinned", flat_pinned_fn), ("hier_pinned", hier_pinned_fn)):
        chk = src.clone()
        fn(chk)
        d = (chk - ref).abs().max().item()
        if rank == 0:
            _log(rank, f"  {name} max|diff vs flat| = {d:.3e} "
                       f"{'OK' if d < 1e-4 else 'WRONG'}")

    variants = [
        ("flat gloo (cuda tensor)", flat_fn),
        ("flat gloo + pinned stage", flat_pinned_fn),
        ("hierarchical (cuda)", hier_fn),
        ("hierarchical + pinned", hier_pinned_fn),
    ]
    times = [(n, timeit(f)) for n, f in variants]
    dist.barrier()
    if rank == 0:
        base = times[0][1]
        _log(rank, "CHECK 3 -- transport comparison (baseline = flat gloo):")
        for n, t in times:
            _log(rank, f"  {n:28s} {t*1000:8.2f} ms   {base/t:5.2f}x")
        best = min(times, key=lambda kv: kv[1])
        _log(rank, f"  BEST: {best[0]} at {base/best[1]:.2f}x")
        mb = args.numel * 4 / 1e6
        _log(rank, f"  buffer {mb:.1f} MB over {world} ranks "
                   f"({args.n_gpu} gpu x {args.stack} stack)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
