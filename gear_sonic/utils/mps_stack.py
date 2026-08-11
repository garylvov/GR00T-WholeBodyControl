"""MPS rank-stacking: N training ranks co-located on each physical GPU.

SONIC at one rank per GPU leaves most of an H100-class node on the floor
(measured on gpu3202/job 4869919: 17.4-18.2 GiB of 95.6 GiB used per GPU, 49-62%
SM utilisation).  Stacking S ranks onto each of the N GPUs behind a single CUDA
MPS daemon recovers the idle fraction.  This module is the Python half; the
shell half (MPS daemon lifecycle, ulimits, allocator config) lives in
``launch_sonic_mps.sh``.

Ported from the ProtoMotions teacher, which runs 8 GPUs x 3 stacked ranks x 8192
envs in production (``pm_run_v62/launch_protomotions_ddp.sh`` and
``protomotions/utils/fabric_config.py``).

Contract -- the launcher is the single source of truth, this module only reads:

    SONIC_NGPU              physical GPUs the launch asked for (default: visible count)
    SONIC_STACK             ranks co-located per GPU           (default: 1)
    SONIC_RANK_STAGGER_SEC  per-rank Kit/PhysX boot stagger    (default: 12 when S>1)
    SONIC_DDP_BACKEND       explicit process-group backend override (optional)

    requested world size == SONIC_NGPU * SONIC_STACK

Why each piece exists:

* **Device round-robin is free, but only for accelerate.**  accelerate's
  ``PartialState.set_device`` assigns
  ``device_index = local_process_index % torch.cuda.device_count()``, so ranks
  land on GPUs 0..N-1, 0..N-1, ... with no patch.  IsaacLab's ``AppLauncher``
  does *not*: under ``distributed=True`` it hard-sets
  ``device_id = int(os.environ["LOCAL_RANK"])`` and then
  ``active_gpu = physics_gpu = device_id``, which is out of range the moment
  world_size > n_gpus (rank 8 of 24 would ask PhysX for cuda:8 on an 8-GPU node).
  :func:`configure_app_launcher_for_stack` takes that branch away from
  AppLauncher and hands it the device accelerate already computed.

* **gloo, not NCCL.**  NCCL's bootstrap refuses two ranks of one communicator on
  one CUDA device ("Duplicate GPU detected"), with or without MPS.  This is the
  same restriction that forced gloo on the teacher's stacked path.

* **Thread caps are mandatory, not tuning.**  ``AppLauncher(distributed=True)``
  is also what normally derives PXR_WORK_THREAD_LIMIT / OPENBLAS_NUM_THREADS and
  the ``carb.tasking`` ``threadCount`` from ``nproc // WORLD_SIZE``.  Disabling
  that branch means we must re-apply them ourselves.  Kit's carb.tasking pool
  sizes itself to hardware concurrency *per rank* and is not bounded by
  OMP/TBB -- it is the load-bearing fork-bomb source (imprint 2026-07-23:
  8 ranks x ~112 carb threads drove gpu3202 to load ~900 and OOM-killed the job).

* **Boot stagger.**  S Kit/PhysX warm-starts hitting one GPU simultaneously
  spikes both the thread count and PhysX tensor memory.  Rank r sleeps
  ``r * SONIC_RANK_STAGGER_SEC`` before AppLauncher.  (Kit init itself is already
  serialised by the ``/tmp/isaaclab_app_launcher.lock`` flock in
  ``train_agent_trl.py``; the stagger keeps the pre-lock CUDA/PhysX warm-up from
  bunching up.)
"""

import os
import time

__all__ = [
    "StackShape",
    "WorldSizeMismatch",
    "stack_shape",
    "assert_world_size_matches_request",
    "resolve_ddp_backend",
    "thread_caps",
    "apply_thread_caps",
    "stagger_boot",
    "configure_app_launcher_for_stack",
    "patch_accelerate_device_assumptions",
    "init_distributed_early",
    "assert_backend_intact",
]

# carb's nested task-wait can deadlock at a hard threadCount=1, so the scheduler
# pool never goes below 2 even when every other thread source is clamped to 1
# (teacher launcher, STACK>=4 branch).
_CARB_MIN_THREADS = 2

# Backend the stacked run must keep, re-asserted every time accelerate rebuilds
# its state. See _force_backend.
_FORCED_BACKEND = None


class WorldSizeMismatch(RuntimeError):
    """Requested rank shape and realised world size disagree.

    ``--num_processes`` is only a *request*: the realised world size comes from
    the launcher's cluster environment, and a launcher under ``srun`` can
    silently collapse to world_size 1 (or to n_gpus) while still printing the
    shape you asked for.  That failure is invisible in the logs -- training just
    runs at a fraction of the intended throughput on a fraction of the node --
    so it is raised, not warned.
    """


class StackShape(tuple):
    """``(n_gpu, stack)`` with the derived world size attached."""

    __slots__ = ()

    def __new__(cls, n_gpu: int, stack: int):
        return super().__new__(cls, (int(n_gpu), int(stack)))

    @property
    def n_gpu(self) -> int:
        return self[0]

    @property
    def stack(self) -> int:
        return self[1]

    @property
    def world_size(self) -> int:
        return self[0] * self[1]

    @property
    def stacked(self) -> bool:
        return self[1] > 1

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"StackShape(n_gpu={self.n_gpu}, stack={self.stack}, world_size={self.world_size})"


def _visible_gpu_count(env=None) -> int:
    """GPU count, preferring CUDA_VISIBLE_DEVICES so the guard tests stay CPU-only."""
    env = os.environ if env is None else env
    cvd = env.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return len([d for d in cvd.split(",") if d.strip() != ""])
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def stack_shape(env=None) -> StackShape:
    """Resolve the requested ``(n_gpu, stack)`` from the environment."""
    env = os.environ if env is None else env
    n_gpu = int(env.get("SONIC_NGPU") or 0) or _visible_gpu_count(env)
    stack = int(env.get("SONIC_STACK") or 1)
    if n_gpu < 1:
        raise WorldSizeMismatch(
            "SONIC_NGPU resolved to 0: no CUDA devices visible and no explicit "
            "SONIC_NGPU set. Refusing to guess the rank shape."
        )
    if stack < 1:
        raise WorldSizeMismatch(f"SONIC_STACK must be >= 1, got {stack!r}")
    return StackShape(n_gpu, stack)


def assert_world_size_matches_request(world_size: int, shape: StackShape = None, env=None) -> StackShape:
    """Fail loudly unless ``world_size == SONIC_NGPU * SONIC_STACK``.

    Call this immediately after the ``Accelerator`` is built, with
    ``accelerator.num_processes``.  See :class:`WorldSizeMismatch` for why this
    is a hard failure.
    """
    env = os.environ if env is None else env
    shape = stack_shape(env) if shape is None else shape
    world_size = int(world_size)
    if world_size != shape.world_size:
        raise WorldSizeMismatch(
            f"rank-shape guard: requested SONIC_NGPU={shape.n_gpu} x "
            f"SONIC_STACK={shape.stack} = {shape.world_size} ranks, but the realised "
            f"world size is {world_size}. `--num_processes` is only a request -- the "
            "world size comes from the launcher's cluster environment. Check that "
            "`accelerate launch --num_processes` matches SONIC_NGPU*SONIC_STACK and "
            "that nothing (srun task count, an accelerate config file, "
            "ACCELERATE_* env) is overriding it. Refusing to train on a silently "
            "collapsed world."
        )
    return shape


def resolve_ddp_backend(shape: StackShape = None, env=None) -> str:
    """Process-group backend for this rank shape.

    ``gloo`` whenever ranks are co-located: NCCL rejects two ranks of one
    communicator on one CUDA device ("Duplicate GPU detected"), MPS included.
    """
    env = os.environ if env is None else env
    override = env.get("SONIC_DDP_BACKEND")
    if override:
        return override
    shape = stack_shape(env) if shape is None else shape
    return "gloo" if shape.stacked else "nccl"


def thread_caps(world_size: int, n_cpu: int = None) -> dict:
    """Per-rank CPU-thread caps for a ``world_size``-way co-located launch.

    Mirrors what ``AppLauncher(distributed=True)`` would have done
    (``nproc // WORLD_SIZE``) and adds the carb floor.  Returns the env mapping
    plus ``carb`` (the ``carb.tasking`` ``threadCount``, which is a Kit CLI arg,
    not an env var).
    """
    if n_cpu is None:
        # sched_getaffinity, not cpu_count: under Slurm the cgroup gives us a
        # slice of the node and cpu_count() would report all 128 logical cores.
        try:
            n_cpu = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            n_cpu = os.cpu_count()
    world_size = max(int(world_size), 1)
    per_rank = max((n_cpu or 1) // world_size, 1)
    return {
        "OMP_NUM_THREADS": str(per_rank),
        "OPENBLAS_NUM_THREADS": str(per_rank),
        "MKL_NUM_THREADS": str(per_rank),
        "NUMEXPR_NUM_THREADS": str(per_rank),
        "TBB_THREAD_COUNT": str(per_rank + 1),
        "PXR_WORK_THREAD_LIMIT": str(per_rank),
        "carb": max(per_rank, _CARB_MIN_THREADS),
    }


def apply_thread_caps(world_size: int, n_cpu: int = None, env=None) -> dict:
    """Export :func:`thread_caps` into the environment; return the mapping."""
    env = os.environ if env is None else env
    caps = thread_caps(world_size, n_cpu=n_cpu)
    for key, value in caps.items():
        if key != "carb":
            env[key] = value
    return caps


def stagger_boot(local_rank: int, shape: StackShape = None, env=None, sleep=time.sleep) -> float:
    """Sleep ``local_rank * SONIC_RANK_STAGGER_SEC`` before Kit/PhysX warm-up."""
    env = os.environ if env is None else env
    shape = stack_shape(env) if shape is None else shape
    default = "12" if shape.stacked else "0"
    seconds = float(env.get("SONIC_RANK_STAGGER_SEC", default)) * int(local_rank)
    if seconds > 0:
        sleep(seconds)
    return seconds


def _force_backend(backend: str) -> None:
    """Make ``backend`` accelerate's default for every (re)construction of state.

    Setting the backend once is not enough.  transformers'
    ``TrainingArguments._setup_devices`` calls
    ``AcceleratorState._reset_state(reset_partial_state=True)`` and then rebuilds
    ``PartialState(backend=self.ddp_backend)`` with ``ddp_backend=None``, which
    ``_prepare_backend`` resolves to ``nccl``.  The *process group* stays gloo
    (it is already initialised, so ``init_process_group`` is skipped), but
    ``PartialState().backend`` now reads ``nccl`` -- and accelerate branches on
    that string, not on reality: ``utils.operations._gpu_gather`` then takes the
    ``all_gather_into_tensor`` path, which gloo does not support, and the run
    dies inside ``trainer.train()`` with
    ``ProcessGroupGloo::allgather: invalid tensor size at index 0``
    (observed gpu3202, 2x2 stack, 2026-08-11).

    So we patch the resolver itself: any construction that does not name a
    backend gets ours.
    """
    global _FORCED_BACKEND
    _FORCED_BACKEND = backend

    from accelerate.state import PartialState

    if getattr(PartialState, "_sonic_backend_forced", False):
        return
    original = PartialState._prepare_backend

    def _prepare_backend(self, cpu: bool = False, sagemaker_dp=False, backend: str = None):
        return original(self, cpu, sagemaker_dp, backend if backend is not None else _FORCED_BACKEND)

    PartialState._prepare_backend = _prepare_backend
    PartialState._sonic_backend_forced = True


def init_distributed_early(shape: StackShape = None, env=None) -> str:
    """Claim accelerate's ``PartialState`` singleton with the right backend.

    MUST be called before anything else touches accelerate -- in particular
    before ``HfArgumentParser.parse_dict`` builds a ``PPOConfig``, because
    transformers' ``TrainingArguments.__post_init__`` resolves ``self.device``,
    which constructs ``PartialState()`` with the default backend.  ``PartialState``
    is a *singleton*: the first construction wins and every later
    ``InitProcessGroupKwargs(backend=...)`` is silently ignored.

    Symptom when this is skipped, with the backend request quietly dropped:
    ``ncclInvalidUsage ... Duplicate GPU detected : rank 3 and rank 1 both on
    CUDA device 8f000`` at the first barrier (observed gpu3202, 2x2 stack) --
    the run had *logged* backend=gloo while actually running NCCL.

    Returns the backend the process group was initialised with.
    """
    env = os.environ if env is None else env
    shape = stack_shape(env) if shape is None else shape
    backend = resolve_ddp_backend(shape, env)
    if not shape.stacked:
        # Unstacked runs keep the stock ordering byte for byte: transformers
        # builds PartialState with nccl exactly as before.
        return backend
    patch_accelerate_device_assumptions(shape, env)

    from accelerate.state import PartialState

    state = PartialState(backend=backend)
    actual = getattr(state, "backend", None)
    if shape.stacked and actual != backend:
        raise WorldSizeMismatch(
            f"backend guard: asked for '{backend}' (mandatory for {shape.stack} "
            f"co-located ranks/GPU -- NCCL rejects them with 'Duplicate GPU "
            f"detected') but accelerate's PartialState is already initialised "
            f"with '{actual}'. Something constructed PartialState before "
            "init_distributed_early(); move the call earlier."
        )
    return backend


def assert_backend_intact(shape: StackShape = None, env=None) -> str:
    """Re-assert the backend after accelerate's state has been rebuilt.

    Checks BOTH the real process group and the string accelerate branches on --
    they can disagree (see :func:`_force_backend`), and a disagreement is a
    crash hours later inside training, not at startup.
    """
    env = os.environ if env is None else env
    shape = stack_shape(env) if shape is None else shape
    want = resolve_ddp_backend(shape, env)

    import torch
    from accelerate.state import PartialState

    real = torch.distributed.get_backend() if torch.distributed.is_initialized() else None
    seen = getattr(PartialState(), "backend", None)
    if shape.stacked and (real != want or seen != want):
        raise WorldSizeMismatch(
            f"backend guard: {shape.stack} ranks/GPU require '{want}', but the "
            f"process group is '{real}' and accelerate believes '{seen}'. NCCL "
            "rejects co-located ranks ('Duplicate GPU detected'), and a "
            "gloo group that accelerate thinks is nccl crashes in "
            "_gpu_gather (all_gather_into_tensor is unsupported on gloo)."
        )
    return want


def patch_accelerate_device_assumptions(shape: StackShape = None, env=None) -> bool:
    """Undo accelerate's two ``local_process_index IS the CUDA ordinal`` assumptions.

    accelerate resolves the *device* correctly for stacked ranks
    (``local_process_index % device_count``), but two other call sites hand the
    raw local rank to CUDA, which is out of range as soon as world_size > n_gpus:

    1. ``PartialState.wait_for_everyone`` -> ``torch.distributed.barrier(
       device_ids=[self.local_process_index])``.  ``barrier`` turns that into
       ``torch.device("cuda", 23)`` and dies with
       ``RuntimeError: CUDA error: invalid device ordinal``.  Observed on
       gpu3202 with a 2x2 stack: local_rank 2 on a 2-GPU view.  ``device_ids`` is
       a NCCL-only hint anyway and we are on gloo when stacked, so we drop it.

    2. ``Accelerator.prepare_model`` -> ``DistributedDataParallel(
       device_ids=[self.local_process_index], output_device=...)``.  accelerate
       already has an escape hatch for this one: ``ACCELERATE_BYPASS_DEVICE_MAP=true``
       makes it pass ``device_ids=None``, and DDP then infers the (already
       correct) device from the module's parameters.

    No-op when not stacked, and idempotent.  Returns True if patches were applied.
    """
    env = os.environ if env is None else env
    shape = stack_shape(env) if shape is None else shape
    if not shape.stacked:
        return False

    env["ACCELERATE_BYPASS_DEVICE_MAP"] = "true"
    _force_backend(resolve_ddp_backend(shape, env))

    from accelerate.state import PartialState

    if getattr(PartialState, "_sonic_mps_stack_patched", False):
        return True

    import torch

    def wait_for_everyone(self):
        # device_ids intentionally omitted: see (1) above.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    PartialState.wait_for_everyone = wait_for_everyone
    PartialState._sonic_mps_stack_patched = True
    return True


def configure_app_launcher_for_stack(args_cli, device: str, world_size: int, shape: StackShape = None, env=None):
    """Point IsaacLab's ``AppLauncher`` at the accelerate-assigned GPU.

    Only touches ``args_cli`` when ranks are actually stacked, so the
    one-rank-per-GPU path keeps IsaacLab's stock ``distributed=True`` behaviour
    byte for byte.

    Returns the applied thread caps (``{}`` when not stacked).
    """
    env = os.environ if env is None else env
    shape = stack_shape(env) if shape is None else shape
    if not shape.stacked:
        return {}

    # AppLauncher's distributed branch would overwrite device with
    # cuda:$LOCAL_RANK (out of range past n_gpus) and set active_gpu/physics_gpu
    # from it. Disable the branch and pass the device accelerate resolved
    # (local_process_index % device_count) instead.
    args_cli.distributed = False
    args_cli.multi_gpu = False
    args_cli.device = device

    # ...which also means AppLauncher no longer caps threads for us.
    caps = apply_thread_caps(world_size, env=env)
    thread_arg = f"--/plugins/carb.tasking.plugin/threadCount={caps['carb']}"
    existing = getattr(args_cli, "kit_args", "") or ""
    if "carb.tasking.plugin/threadCount" not in existing:
        args_cli.kit_args = f"{existing} {thread_arg}".strip()
    return caps
