# SPDX-License-Identifier: Apache-2.0
"""Guard tests for the hierarchical all-reduce rank arithmetic.

CPU-only and distributed-free by construction: the rank maths is where the
bugs that hang a 24-rank job actually live, and a test that needs 8 GPUs to
run is a test nobody runs.
"""

import pytest

from gear_sonic.utils.hierarchical_pg import (
    HierarchyError,
    device_of_rank,
    intra_ranks,
    is_leader,
    leader_of_device,
    leader_ranks,
    partition,
)


# --------------------------------------------------------------------------
# The invariant that makes the NCCL level legal at all.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_gpu,stack", [(8, 3), (8, 2), (4, 4), (2, 3), (1, 5), (8, 1)])
def test_layout_is_a_partition(n_gpu, stack):
    groups = partition(n_gpu, stack)
    assert len(groups) == n_gpu
    assert all(len(g) == stack for g in groups)
    flat = sorted(r for g in groups for r in g)
    assert flat == list(range(n_gpu * stack))


@pytest.mark.parametrize("n_gpu,stack", [(8, 3), (8, 2), (4, 4), (2, 3)])
def test_no_two_leaders_share_a_device(n_gpu, stack):
    """The exact condition NCCL rejected on the flat group."""
    leaders = leader_ranks(n_gpu)
    devices = [device_of_rank(r, n_gpu) for r in leaders]
    assert len(set(devices)) == n_gpu, (
        f"leaders {leaders} -> devices {devices}; a repeat here is the "
        '"Duplicate GPU detected" failure that forced gloo in the first place'
    )


def test_production_shape_layout_is_exactly_as_documented():
    """8x3 is the shape we run. Pin its layout literally, not derivationally."""
    groups = partition(8, 3)
    assert groups[0] == [0, 8, 16]
    assert groups[1] == [1, 9, 17]
    assert groups[7] == [7, 15, 23]
    assert leader_ranks(8) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_leader_is_the_first_member_of_its_own_intra_group():
    """The hook broadcasts from `leader` inside `intra_pg`; if the leader were
    not a member of that group the broadcast would raise at runtime."""
    for n_gpu, stack in [(8, 3), (4, 2), (2, 5)]:
        for d in range(n_gpu):
            members = intra_ranks(d, n_gpu, stack)
            leader = leader_of_device(d, n_gpu)
            assert leader in members
            assert members[0] == leader


def test_every_rank_maps_to_its_own_groups_device():
    for n_gpu, stack in [(8, 3), (3, 3), (5, 2)]:
        for d in range(n_gpu):
            for r in intra_ranks(d, n_gpu, stack):
                assert device_of_rank(r, n_gpu) == d


@pytest.mark.parametrize("n_gpu,stack", [(8, 3), (4, 2)])
def test_is_leader_agrees_with_leader_ranks(n_gpu, stack):
    for r in range(n_gpu * stack):
        assert is_leader(r, n_gpu) == (r in leader_ranks(n_gpu))


# --------------------------------------------------------------------------
# Failures must be loud. A silent fallback restores the 7 s/iter gloo cost.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1])
def test_zero_or_negative_gpu_count_raises(bad):
    with pytest.raises(HierarchyError):
        device_of_rank(0, bad)


def test_negative_rank_raises():
    with pytest.raises(HierarchyError):
        device_of_rank(-1, 8)


@pytest.mark.parametrize("device", [-1, 8, 99])
def test_out_of_range_device_raises(device):
    with pytest.raises(HierarchyError):
        intra_ranks(device, 8, 3)
    with pytest.raises(HierarchyError):
        leader_of_device(device, 8)


@pytest.mark.parametrize("bad", [0, -3])
def test_non_positive_stack_raises(bad):
    with pytest.raises(HierarchyError):
        intra_ranks(0, 8, bad)


def test_install_refuses_unstacked_shape():
    """stack=1 has no co-location, so the hierarchy is pure overhead and the
    plain NCCL path is already correct. Refuse rather than silently degrade."""
    from gear_sonic.utils.hierarchical_pg import install_hierarchical_allreduce

    with pytest.raises(HierarchyError, match="not co-located"):
        install_hierarchical_allreduce(object(), n_gpu=8, stack=1)


def test_install_refuses_a_model_with_no_ddp_wrapper():
    """A warn-and-continue here would leave the flat gloo all-reduce in place
    and the only symptom would be a slow run -- exactly the class of silent
    non-measurement this campaign kept getting burned by."""
    from gear_sonic.utils.hierarchical_pg import install_hierarchical_allreduce

    class Plain:
        pass

    with pytest.raises(HierarchyError, match="no DistributedDataParallel"):
        install_hierarchical_allreduce(Plain(), n_gpu=8, stack=3)


def test_build_groups_requires_an_initialized_process_group():
    from gear_sonic.utils.hierarchical_pg import build_hierarchical_groups

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    with pytest.raises(HierarchyError, match="not initialized"):
        build_hierarchical_groups(8, 3)


# --------------------------------------------------------------------------
# Reduction algebra: the hook must produce a plain mean, like DDP's default.
# --------------------------------------------------------------------------

def test_three_level_reduction_equals_a_flat_mean():
    """Simulate intra-sum -> inter-sum -> broadcast -> /world and check it
    equals the flat mean DDP's default hook would have produced. A missing
    div_ shows up as a world_size-times-too-large gradient, which reads as a
    diverging learning rate rather than as a comms bug."""
    n_gpu, stack = 8, 3
    world = n_gpu * stack
    grads = {r: float(r + 1) for r in range(world)}

    intra_sum = {}
    for d in range(n_gpu):
        members = intra_ranks(d, n_gpu, stack)
        s = sum(grads[r] for r in members)
        for r in members:
            intra_sum[r] = s

    leader_total = sum(intra_sum[leader_of_device(d, n_gpu)] for d in range(n_gpu))

    final = {}
    for d in range(n_gpu):
        for r in intra_ranks(d, n_gpu, stack):
            final[r] = leader_total / world

    expected = sum(grads.values()) / world
    assert all(abs(v - expected) < 1e-9 for v in final.values())
    assert len(set(final.values())) == 1, "all ranks must end with the same value"
