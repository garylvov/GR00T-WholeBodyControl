# SPDX-License-Identifier: Apache-2.0
"""Hierarchical gradient all-reduce for MPS-stacked ranks.

STATUS: MEASURED AND REFUTED IN THIS FORM.  DO NOT ENABLE.  READ THIS FIRST.
---------------------------------------------------------------------------
The gloo-intra / NCCL-inter design below is CORRECT and SLOWER.  Measured on
gpu3202 with a 32 MB buffer, via ``smoke_hierarchical_pg.py``::

    shape           flat gloo      hierarchical      ratio
    2 gpu x 2        23.48 ms         29.97 ms       0.78x
    8 gpu x 3        66.13 ms         90.54 ms       0.73x   <- production shape

Checks 1 and 2 PASS and are worth keeping: a leaders-only NCCL subgroup DOES
construct while the default group is gloo with co-located ranks (this was the
open question -- it is answered, and the answer is yes), and the three-level
reduce is numerically identical to a flat all-reduce (max|diff| = 2.98e-07).

Check 3 fails because the cost model behind the design was wrong.  Gloo's
expense on CUDA tensors is the **device-to-host-to-device copy**, not the ring
topology.  Shrinking the ring from 24 members to 3 therefore saves almost
nothing, while splitting one collective into two (intra all-reduce, then
broadcast down) pays that copy TWICE.  The NCCL leg is nearly free by
comparison and cannot make up the difference.

What this rules IN, for whoever picks it up: the only variant that can win is
one with ZERO host copies on the intra-GPU leg -- a CUDA IPC shared buffer and
a device-local reduction kernel, then NCCL across the leaders, then an IPC
copy back down.  That preserves the gradient math exactly (same sum, same
order-independence tolerance) while removing the term that actually costs.
The rank arithmetic, group construction, and hook plumbing in this module are
all reusable for that variant; only the two ``dist`` calls on ``intra_pg``
need replacing.

Keep this module.  A refuted design with its measurement attached is worth
more than an untried idea, because it stops the next person re-deriving it.

WHY IT WAS BUILT
----------------
Co-locating ``stack`` ranks per GPU under MPS wins on the simulation side --
measured 555k -> 925k env-steps/s collection, 1.67x -- but NCCL's bootstrap
refuses two ranks of one communicator on one device ("Duplicate GPU
detected"), so a flat 24-rank group must fall back to gloo.  Gloo's all-reduce
is host-side: gradients leave the GPU, are summed on the CPU, and come back.
Measured at 8x3x4096 that cost **7.06 s of a 9.62 s iteration** and turned a
1.67x collection win into a 21% net throughput LOSS.  Widening gloo transport
(4 socket threads x 2 socks) changed nothing, which localises the cost to the
host reduction rather than socket width.

The flat group is the problem, not gloo itself.  Decompose it:

    level 1   the ``stack`` ranks sharing one device        gloo, 2 hops
    level 2   the ``n_gpu`` leaders, one per device         NCCL  <-- legal!
    level 3   leaders push the result back down             gloo, 2 hops

Level 2 is legal NCCL precisely because the inter-GPU group holds exactly one
rank per device -- the condition NCCL requires and the flat group violated.
It is the same 8-way path the unstacked baseline runs at 1.03 s.  Level 1
shrinks the gloo ring from 24 members to ``stack`` (3), and it runs once per
GPU rather than across the whole job.

RANK LAYOUT
-----------
accelerate's ``PartialState.set_device`` assigns
``device_index = local_process_index % torch.cuda.device_count()``, so for
``n_gpu=8, stack=3`` the layout is::

    GPU 0 <- ranks 0, 8, 16        GPU 4 <- ranks 4, 12, 20
    GPU 1 <- ranks 1, 9, 17        GPU 5 <- ranks 5, 13, 21
    ...                            GPU 7 <- ranks 7, 15, 23

Ranks ``0..n_gpu-1`` are therefore one-per-device already, and are used as
leaders unchanged.  Do not "improve" this to a blocked layout without changing
:func:`device_of_rank` with it -- the two must agree or the NCCL group will be
handed two ranks on one device and fail at construction.

WHAT THIS DOES NOT FIX
----------------------
Throughput is not the objective; optimizer steps are.  ``num_mini_batches``
and ``num_learning_epochs`` are fixed at 4 and 5, so a rank does 20 optimizer
steps per iteration REGARDLESS of ``num_envs``.  Raising envs per rank raises
the minibatch size and lowers gradient steps per wall-second.  Hold the
minibatch size constant by scaling ``num_mini_batches`` with
``num_envs`` -- see ``docs`` and the companion note in ``launch_sonic_mps.sh``.
This module makes the stacked shape affordable; it does not make it correct.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

__all__ = [
    "device_of_rank",
    "intra_ranks",
    "leader_of_device",
    "is_leader",
    "leader_ranks",
    "partition",
    "HierarchicalGroups",
    "build_hierarchical_groups",
    "make_hierarchical_hook",
    "install_hierarchical_allreduce",
    "HierarchyError",
]


class HierarchyError(RuntimeError):
    """Raised when a rank layout cannot support a hierarchical group.

    Always a hard error.  A silent fall back to the flat gloo group would
    reintroduce exactly the 7 s/iter cost this module exists to remove, and it
    would do so invisibly -- the run would simply be slow.
    """


# --------------------------------------------------------------------------
# Pure rank arithmetic.  No torch import needed, so this half is unit-testable
# on a CPU box with no distributed context -- which is where its bugs live.
# --------------------------------------------------------------------------

def device_of_rank(rank: int, n_gpu: int) -> int:
    """Device index a global ``rank`` lands on. Mirrors accelerate's modulo."""
    if n_gpu <= 0:
        raise HierarchyError(f"n_gpu must be >= 1, got {n_gpu!r}")
    if rank < 0:
        raise HierarchyError(f"rank must be >= 0, got {rank!r}")
    return rank % n_gpu


def intra_ranks(device: int, n_gpu: int, stack: int) -> List[int]:
    """Global ranks co-located on ``device``, leader first."""
    if not (0 <= device < n_gpu):
        raise HierarchyError(f"device {device} out of range for n_gpu={n_gpu}")
    if stack <= 0:
        raise HierarchyError(f"stack must be >= 1, got {stack!r}")
    return [device + i * n_gpu for i in range(stack)]


def leader_of_device(device: int, n_gpu: int) -> int:
    """Global rank that speaks for ``device`` in the inter-GPU group."""
    if not (0 <= device < n_gpu):
        raise HierarchyError(f"device {device} out of range for n_gpu={n_gpu}")
    return device


def leader_ranks(n_gpu: int) -> List[int]:
    """The inter-GPU group: exactly one rank per device, which NCCL requires."""
    return list(range(n_gpu))


def is_leader(rank: int, n_gpu: int) -> bool:
    return rank < n_gpu


def partition(n_gpu: int, stack: int) -> List[List[int]]:
    """Full layout as ``[[ranks on gpu 0], [ranks on gpu 1], ...]``.

    Verifies the invariant that makes the NCCL level legal: every rank appears
    exactly once, and no two leaders share a device.
    """
    groups = [intra_ranks(d, n_gpu, stack) for d in range(n_gpu)]
    flat = [r for g in groups for r in g]
    if sorted(flat) != list(range(n_gpu * stack)):
        raise HierarchyError(
            f"layout for n_gpu={n_gpu} stack={stack} is not a partition of "
            f"0..{n_gpu * stack - 1}: got {sorted(flat)}"
        )
    leaders = [g[0] for g in groups]
    devices = [device_of_rank(r, n_gpu) for r in leaders]
    if len(set(devices)) != len(devices):
        raise HierarchyError(
            f"leaders {leaders} map to devices {devices} -- two leaders share a "
            "device, so the inter-GPU NCCL group would be rejected"
        )
    return groups


# --------------------------------------------------------------------------
# Process-group construction and the DDP comm hook.
# --------------------------------------------------------------------------

class HierarchicalGroups:
    """The two process groups plus the local rank's place in them."""

    def __init__(self, rank, n_gpu, stack, intra_pg, inter_pg, device, leader):
        self.rank = rank
        self.n_gpu = n_gpu
        self.stack = stack
        self.intra_pg = intra_pg
        self.inter_pg = inter_pg      # None on non-leaders: they are not members
        self.device = device
        self.leader = leader          # global rank of this device's leader
        self.world_size = n_gpu * stack

    @property
    def is_leader(self) -> bool:
        return self.rank == self.leader

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"HierarchicalGroups(rank={self.rank} device={self.device} "
            f"leader={self.leader} is_leader={self.is_leader} "
            f"n_gpu={self.n_gpu} stack={self.stack})"
        )


def build_hierarchical_groups(n_gpu: int, stack: int, rank: Optional[int] = None):
    """Create the intra-GPU (gloo) and inter-GPU (NCCL) groups.

    ``new_group`` is collective: EVERY rank must call it for EVERY group, in
    the same order, including groups it is not a member of.  Non-members get
    back ``GroupMember.NON_GROUP_MEMBER`` and must never pass it to a
    collective.  Getting this wrong hangs the job at startup rather than
    failing, which is why the loop below is unconditional and the membership
    test happens after.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        raise HierarchyError("torch.distributed is not initialized")

    partition(n_gpu, stack)  # validate before building anything
    rank = dist.get_rank() if rank is None else rank
    world = dist.get_world_size()
    if world != n_gpu * stack:
        raise HierarchyError(
            f"world_size {world} != n_gpu*stack {n_gpu * stack}; refusing to "
            "build a hierarchy over a shape that is not the one running"
        )

    my_device = device_of_rank(rank, n_gpu)
    my_leader = leader_of_device(my_device, n_gpu)

    intra_pg = None
    for d in range(n_gpu):
        members = intra_ranks(d, n_gpu, stack)
        g = dist.new_group(ranks=members, backend="gloo")
        if rank in members:
            intra_pg = g
    if intra_pg is None:
        raise HierarchyError(f"rank {rank} joined no intra-GPU group")

    # Collective on ALL ranks; only the leaders end up as members.
    inter_grp = dist.new_group(ranks=leader_ranks(n_gpu), backend="nccl")
    inter_pg = inter_grp if is_leader(rank, n_gpu) else None

    return HierarchicalGroups(
        rank=rank, n_gpu=n_gpu, stack=stack,
        intra_pg=intra_pg, inter_pg=inter_pg,
        device=my_device, leader=my_leader,
    )


def make_hierarchical_hook(groups: HierarchicalGroups):
    """DDP communication hook implementing the three-level reduce.

    DDP would otherwise call ``all_reduce`` on the flat group.  The hook
    replaces that with intra-sum / inter-sum / broadcast-down, then applies the
    ``1/world_size`` scaling DDP's default hook applies -- omit it and every
    gradient is ``world_size`` times too large, which looks like a diverging
    learning rate rather than a comms bug.
    """
    import torch
    import torch.distributed as dist

    def hook(state, bucket):
        buf = bucket.buffer()

        # 1. sum the ranks sharing this device (gloo, `stack` members)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=groups.intra_pg)

        # 2. leaders only: sum across devices (NCCL, one rank per device)
        if groups.inter_pg is not None:
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=groups.inter_pg)

        # 3. push the global sum back to this device's non-leaders
        dist.broadcast(buf, src=groups.leader, group=groups.intra_pg)

        buf.div_(groups.world_size)

        fut = torch.futures.Future()
        fut.set_result(buf)
        return fut

    return hook


def _unwrap_to_ddp(model):
    """Find the DistributedDataParallel wrapper accelerate produced."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    seen = 0
    cur = model
    while cur is not None and seen < 8:
        if isinstance(cur, DDP):
            return cur
        cur = getattr(cur, "module", None)
        seen += 1
    return None


def install_hierarchical_allreduce(model, n_gpu: int, stack: int):
    """Register the hierarchical hook on ``model``. Returns the groups.

    Raises rather than warning when the model is not DDP-wrapped: a silent skip
    leaves the flat gloo all-reduce in place, which is the slow path this exists
    to replace, and the only symptom would be a slow run.
    """
    if stack <= 1:
        raise HierarchyError(
            f"stack={stack} is not co-located; use the plain NCCL path "
            "(resolve_ddp_backend already returns 'nccl' for it)"
        )

    ddp = _unwrap_to_ddp(model)
    if ddp is None:
        raise HierarchyError(
            f"no DistributedDataParallel found on {type(model).__name__}; "
            "cannot install a comm hook, and falling through would silently "
            "keep the flat gloo all-reduce"
        )

    groups = build_hierarchical_groups(n_gpu, stack)
    ddp.register_comm_hook(state=None, hook=make_hierarchical_hook(groups))
    print(
        f"[hierarchical_pg] installed: {groups!r} "
        f"(intra=gloo/{stack}, inter=nccl/{n_gpu})",
        flush=True,
    )
    return groups
