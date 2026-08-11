"""CPU-only guard tests for MPS rank-stacking (``gear_sonic.utils.mps_stack``).

The load-bearing test is :func:`test_world_size_mismatch_raises`: ``--num_processes``
is only a *request*, the realised world size comes from the launcher's cluster
environment, and a silent collapse (to 1, or to n_gpus) has bitten this campaign
twice.  It must fail the run, not warn.

Run:  pytest gear_sonic/tests/test_mps_stack.py
"""

import types

import pytest

from gear_sonic.utils import mps_stack


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "SONIC_NGPU",
        "SONIC_STACK",
        "SONIC_RANK_STAGGER_SEC",
        "SONIC_DDP_BACKEND",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(key, raising=False)


def _env(**kw):
    return {str(k): str(v) for k, v in kw.items()}


# --------------------------------------------------------------- shape resolution


def test_shape_from_explicit_env():
    shape = mps_stack.stack_shape(_env(SONIC_NGPU=8, SONIC_STACK=3))
    assert (shape.n_gpu, shape.stack, shape.world_size) == (8, 3, 24)
    assert shape.stacked


def test_shape_defaults_to_unstacked():
    shape = mps_stack.stack_shape(_env(SONIC_NGPU=8))
    assert shape.world_size == 8
    assert not shape.stacked


def test_shape_falls_back_to_visible_devices():
    shape = mps_stack.stack_shape(_env(CUDA_VISIBLE_DEVICES="0,1,2,3", SONIC_STACK=2))
    assert (shape.n_gpu, shape.world_size) == (4, 8)


def test_shape_rejects_zero_stack():
    with pytest.raises(mps_stack.WorldSizeMismatch):
        mps_stack.stack_shape(_env(SONIC_NGPU=8, SONIC_STACK=0))


def test_shape_refuses_to_guess_without_gpus():
    with pytest.raises(mps_stack.WorldSizeMismatch):
        mps_stack.stack_shape(_env(CUDA_VISIBLE_DEVICES="", SONIC_NGPU=0, SONIC_STACK=3))


# ------------------------------------------------------------------- the guard


def test_world_size_matches_request():
    env = _env(SONIC_NGPU=8, SONIC_STACK=3)
    assert mps_stack.assert_world_size_matches_request(24, env=env).world_size == 24


@pytest.mark.parametrize("realised", [1, 8, 23, 25])
def test_world_size_mismatch_raises(realised):
    """The exact silent-collapse class this guard exists for."""
    env = _env(SONIC_NGPU=8, SONIC_STACK=3)
    with pytest.raises(mps_stack.WorldSizeMismatch) as exc:
        mps_stack.assert_world_size_matches_request(realised, env=env)
    assert "24" in str(exc.value) and str(realised) in str(exc.value)


def test_unstacked_world_size_still_guarded():
    env = _env(SONIC_NGPU=8, SONIC_STACK=1)
    mps_stack.assert_world_size_matches_request(8, env=env)
    with pytest.raises(mps_stack.WorldSizeMismatch):
        mps_stack.assert_world_size_matches_request(1, env=env)


# ----------------------------------------------------------------- ddp backend


def test_backend_is_gloo_when_stacked():
    # NCCL refuses two ranks of one communicator on one CUDA device.
    assert mps_stack.resolve_ddp_backend(env=_env(SONIC_NGPU=8, SONIC_STACK=3)) == "gloo"


def test_backend_is_nccl_when_unstacked():
    assert mps_stack.resolve_ddp_backend(env=_env(SONIC_NGPU=8, SONIC_STACK=1)) == "nccl"


def test_backend_override_wins():
    env = _env(SONIC_NGPU=8, SONIC_STACK=3, SONIC_DDP_BACKEND="nccl")
    assert mps_stack.resolve_ddp_backend(env=env) == "nccl"


# ----------------------------------------------------------------- thread caps


def test_thread_caps_divide_the_node():
    caps = mps_stack.thread_caps(24, n_cpu=96)
    assert caps["OMP_NUM_THREADS"] == "4"
    assert caps["OPENBLAS_NUM_THREADS"] == "4"
    assert caps["PXR_WORK_THREAD_LIMIT"] == "4"
    assert caps["TBB_THREAD_COUNT"] == "5"
    assert caps["carb"] == 4


def test_thread_caps_never_reach_zero_and_carb_has_a_floor():
    # 40 ranks on 96 cores -> 2 threads; a hard carb=1 can deadlock carb's
    # nested task-wait, hence the floor.
    caps = mps_stack.thread_caps(200, n_cpu=96)
    assert caps["OMP_NUM_THREADS"] == "1"
    assert caps["carb"] == mps_stack._CARB_MIN_THREADS >= 2


def test_apply_thread_caps_exports_but_not_carb():
    env = {}
    caps = mps_stack.apply_thread_caps(24, n_cpu=96, env=env)
    assert env["OMP_NUM_THREADS"] == "4"
    assert "carb" not in env  # carb is a Kit CLI arg, not an env var
    assert caps["carb"] == 4


# ------------------------------------------------------- AppLauncher wiring


def _args_cli():
    return types.SimpleNamespace(
        distributed=True, multi_gpu=True, device="cuda:0", kit_args="--/log/level=error"
    )


def test_app_launcher_untouched_when_unstacked():
    args = _args_cli()
    caps = mps_stack.configure_app_launcher_for_stack(
        args, "cuda:3", 8, env=_env(SONIC_NGPU=8, SONIC_STACK=1)
    )
    assert caps == {}
    assert args.distributed is True and args.multi_gpu is True
    assert args.device == "cuda:0"  # stock IsaacLab path left byte-identical


def test_app_launcher_gets_accelerate_device_when_stacked():
    args = _args_cli()
    env = _env(SONIC_NGPU=8, SONIC_STACK=3)
    caps = mps_stack.configure_app_launcher_for_stack(args, "cuda:7", 24, env=env)
    # AppLauncher's distributed branch would have set device_id = LOCAL_RANK,
    # i.e. cuda:23 on an 8-GPU node -> out of range for active_gpu/physics_gpu.
    assert args.distributed is False and args.multi_gpu is False
    assert args.device == "cuda:7"
    assert f"threadCount={caps['carb']}" in args.kit_args
    assert "--/log/level=error" in args.kit_args
    assert env["OMP_NUM_THREADS"] == caps["OMP_NUM_THREADS"]


def test_app_launcher_thread_arg_is_idempotent():
    args = _args_cli()
    env = _env(SONIC_NGPU=8, SONIC_STACK=3)
    mps_stack.configure_app_launcher_for_stack(args, "cuda:1", 24, env=env)
    once = args.kit_args
    mps_stack.configure_app_launcher_for_stack(args, "cuda:1", 24, env=env)
    assert args.kit_args == once


def test_accelerate_device_round_robin_matches_our_expectation():
    """Documents the mapping we rely on from accelerate's PartialState.set_device:
    ``device_index = local_process_index % torch.cuda.device_count()``."""
    n_gpu, stack = 8, 3
    mapping = [rank % n_gpu for rank in range(n_gpu * stack)]
    assert mapping[:8] == list(range(8))
    for gpu in range(n_gpu):
        assert mapping.count(gpu) == stack


# --------------------------------------------------------------------- stagger


def test_stagger_scales_with_local_rank():
    slept = []
    env = _env(SONIC_NGPU=8, SONIC_STACK=3, SONIC_RANK_STAGGER_SEC=12)
    assert mps_stack.stagger_boot(0, env=env, sleep=slept.append) == 0
    assert mps_stack.stagger_boot(5, env=env, sleep=slept.append) == 60
    assert slept == [60]


def test_no_stagger_when_unstacked():
    slept = []
    env = _env(SONIC_NGPU=8, SONIC_STACK=1)
    assert mps_stack.stagger_boot(3, env=env, sleep=slept.append) == 0
    assert slept == []


# ------------------------------------------- accelerate device-assumption patch


def test_patch_is_a_noop_when_unstacked():
    env = _env(SONIC_NGPU=8, SONIC_STACK=1)
    assert mps_stack.patch_accelerate_device_assumptions(env=env) is False
    assert "ACCELERATE_BYPASS_DEVICE_MAP" not in env


def test_patch_sets_ddp_bypass_and_replaces_barrier():
    """accelerate barriers on device_ids=[local_process_index]; rank 23 on an
    8-GPU node is 'CUDA error: invalid device ordinal'."""
    from accelerate.state import PartialState

    original = PartialState.__dict__.get("wait_for_everyone")
    try:
        env = _env(SONIC_NGPU=8, SONIC_STACK=3)
        assert mps_stack.patch_accelerate_device_assumptions(env=env) is True
        assert env["ACCELERATE_BYPASS_DEVICE_MAP"] == "true"

        # the replacement comes from mps_stack, and never passes device_ids
        assert PartialState.wait_for_everyone.__module__ == mps_stack.__name__
        calls = []
        fake_dist = types.SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            barrier=lambda **kw: calls.append(kw),
        )
        import torch

        monkey = torch.distributed
        try:
            torch.distributed = fake_dist
            PartialState.wait_for_everyone(object())
        finally:
            torch.distributed = monkey
        assert calls == [{}]  # no device_ids -> no 'invalid device ordinal'
        # idempotent
        assert mps_stack.patch_accelerate_device_assumptions(env=env) is True
    finally:
        if original is not None:
            PartialState.wait_for_everyone = original
        PartialState._sonic_mps_stack_patched = False


def test_init_distributed_early_is_a_noop_when_unstacked():
    """Unstacked runs must keep the stock ordering: transformers builds
    PartialState itself, with nccl, exactly as before."""
    from accelerate.state import PartialState

    before = dict(PartialState._shared_state)
    assert mps_stack.init_distributed_early(env=_env(SONIC_NGPU=8, SONIC_STACK=1)) == "nccl"
    assert dict(PartialState._shared_state) == before


def test_thread_caps_use_the_cgroup_slice_not_the_whole_node(monkeypatch):
    """os.cpu_count() reports all 128 logical cores even inside a 16-CPU Slurm
    cgroup, which would hand each rank an 8x oversized thread budget."""
    monkeypatch.setattr(mps_stack.os, "sched_getaffinity", lambda _pid: set(range(16)))
    monkeypatch.setattr(mps_stack.os, "cpu_count", lambda: 128)
    assert mps_stack.thread_caps(4)["OMP_NUM_THREADS"] == "4"
