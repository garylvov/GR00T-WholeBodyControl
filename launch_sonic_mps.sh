#!/bin/bash
# ============================================================================
# SONIC MPS-stacked launcher: NGPU physical GPUs x STACK ranks per GPU.
#
#   NGPU=8 STACK=3 NUM_ENVS=4096 ./launch_sonic_mps.sh
#     -> world_size 24, ranks round-robin onto cuda:0..7 three deep, all of them
#        sharing ONE nvidia-cuda-mps-control daemon.
#
# STACK=1 reproduces the stock one-rank-per-GPU launch exactly (no MPS daemon,
# NCCL backend, IsaacLab's own distributed device/thread handling) so this file
# can replace the old sbatch body without changing the unstacked baseline.
#
# WHY: at one rank per GPU, SONIC used 17.4-18.2 GiB of 95.6 GiB and 49-62% SM
# on gpu3202 (job 4869919) -- ~5x memory headroom and ~40% idle compute.
# Sizing at the measured ~18.2 GiB/rank:
#     STACK=3 -> ~54.7 GiB/GPU (56%)   <- chosen; matches the teacher's shape
#     STACK=4 -> ~73.0 GiB/GPU (75%)
#     STACK=5 -> ~91.2 GiB/GPU (93%)   <- no room for fragmentation, don't
#
# Ported from the ProtoMotions teacher (pm_run_v62/launch_protomotions_ddp.sh),
# which runs 8x3x8192 in production. The Python half is gear_sonic/utils/mps_stack.py.
# ============================================================================
set -uo pipefail

# Stacked Isaac ranks fork a lot; raise nproc/nofile to the hard max so they
# cannot fork-lock the node (imprint always-raise-ulimit).
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

HERE="$(dirname "$(readlink -f "$0")")"
ENVD="${ENVD:-/oscar/data/stellex/glvov/glvov-envs/sonic-isaaclab232}"

NGPU="${NGPU:-8}"
STACK="${STACK:-3}"
NUM_ENVS="${NUM_ENVS:-4096}"
WORLD=$((NGPU * STACK))
NPROC="$(nproc)"

EXP="${EXP:-manager/universal_token/all_modes/sonic_h1_2}"
EXP_VAR="${EXP_VAR:-run1_vanilla_mps${STACK}}"
MOTION="${MOTION:-/oscar/data/stellex/glvov/sonic_motions/packs/v58_clean_plus_synth20_7class_v2.pkl}"
MAIN_PORT="${MAIN_PORT:-$((29000 + RANDOM % 2000))}"

if [[ ! -f "$MOTION" ]]; then
  echo "[sonic_mps] ERROR: motion corpus not found: $MOTION" >&2
  echo "[sonic_mps]        refusing to silently fall back to another pack." >&2
  exit 3
fi

# REQUIRED: spack's gcc-runtime-11.5.0 shadows the env's newer libstdc++ and
# Isaac Sim dies on `libcarb.so: version GLIBCXX_3.4.30 not found`.
export LD_LIBRARY_PATH="$ENVD/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
# persistent shared shader cache (cold/warm boot delta is a tracked metric)
export OMNI_CACHE_ROOT="${OMNI_CACHE_ROOT:-/oscar/data/stellex/glvov/glvov-envs/.kit-shader-cache}"
mkdir -p "$OMNI_CACHE_ROOT"

# CUDA allocator: do NOT use expandable_segments:True here. It swaps the caching
# allocator onto the CUDA VMM API, and Isaac's omni.physx.fabric DirectGpuHelper
# takes raw GPU pointers that VMM breaks -> illegal memory access during PhysX
# warm-start, with or without MPS. max_split_size_mb:512 keeps the native
# allocator (Fabric-safe AND MPS-safe) while capping oversized block splitting.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

# ---- device sets -------------------------------------------------------------
# MPS reinterprets a CLIENT's CUDA_VISIBLE_DEVICES *relative to the device order
# the MPS server sees*. Start the daemon on the physical devices, then hand the
# ranks the relative indices 0..NGPU-1. Giving both the same physical list only
# happens to work when that list is already 0..N-1; with e.g. CUDA_VISIBLE_DEVICES=6,7
# the server exposes 2 devices (0,1) and a client asking for 6,7 dies with
# "No CUDA GPUs are available" (reproduced on gpu3202, 2026-08-11).
_INHERITED="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$_INHERITED" ]]; then
  _INHERITED="$(seq -s, 0 $(( $(nvidia-smi -L | wc -l) - 1 )))"
fi
IFS=',' read -r -a _ALL <<< "$_INHERITED"
if (( ${#_ALL[@]} < NGPU )); then
  echo "[sonic_mps] ERROR: NGPU=$NGPU but only ${#_ALL[@]} GPUs visible ($_INHERITED)" >&2
  exit 4
fi
MPS_DEVICES="$(IFS=,; echo "${_ALL[*]:0:NGPU}")"   # physical ids for the daemon
RANK_DEVICES="$(seq -s, 0 $((NGPU - 1)))"          # relative ids for the ranks

# ---- H1-2 PRODUCTION PLANT ---------------------------------------------------
# The canonical stiffen ladder our ProtoMotions teacher trains with. These are
# already the in-file defaults in gear_sonic/envs/manager_env/robots/h1_2.py, but
# they are exported EXPLICITLY so the plant is stated by the launcher and cannot
# drift if that default ever changes -- and so an unrelated PM_ARM_* left in the
# environment cannot quietly downgrade the robot to the unconfigured fallback
# (KP 19.74 / KD 1.26 / effort 40-18-18-19, i.e. half the stiffness and a third
# to a sixth of the torque).
# KNOWN DISCREPANCY, deliberate: KP 40 with the declared armature 0.005 does not
# satisfy SONIC's own KP = armature*w^2 formula (that gives 19.74), and KD 2.0 is
# zeta ~= 3.18 w.r.t. that armature. Kept for comparability with the teacher; see
# the note in robots/h1_2.py. Do not "reconcile" it silently.
export PM_ARM_KP="${PM_ARM_KP:-40}"
export PM_ARM_KD="${PM_ARM_KD:-2.0}"
export PM_ARM_EFFORT_SHOULDER="${PM_ARM_EFFORT_SHOULDER:-120}"   # also feeds shoulder YAW
export PM_ARM_EFFORT_ELBOW="${PM_ARM_EFFORT_ELBOW:-120}"
export PM_ARM_EFFORT_WRIST="${PM_ARM_EFFORT_WRIST:-30}"
# Read the gains back out of the live articulation and abort if they are not the
# production plant (proof lands in <experiment_dir>/plant.yaml).
export SONIC_REQUIRE_PRODUCTION_PLANT="${SONIC_REQUIRE_PRODUCTION_PLANT:-1}"

# ---- contract read by gear_sonic/utils/mps_stack.py (single source of truth) --
export SONIC_NGPU="$NGPU"
export SONIC_STACK="$STACK"
export SONIC_RANK_STAGGER_SEC="${SONIC_RANK_STAGGER_SEC:-12}"

# ---- CPU-thread caps ---------------------------------------------------------
# mps_stack re-applies these per rank; setting them here too means even the
# libraries imported before Accelerator() (numpy/BLAS at import time) are capped.
# Kit's carb.tasking pool is the load-bearing fork-bomb source: it sizes to
# hardware concurrency PER RANK and is not bounded by OMP/TBB (imprint
# 2026-07-23: 8 ranks x ~112 carb threads -> load ~900, OOM-kill on gpu3202).
TCAP=$(( NPROC / WORLD >= 1 ? NPROC / WORLD : 1 ))
export OMP_NUM_THREADS=$TCAP OPENBLAS_NUM_THREADS=$TCAP MKL_NUM_THREADS=$TCAP \
       NUMEXPR_NUM_THREADS=$TCAP PXR_WORK_THREAD_LIMIT=$TCAP TBB_THREAD_COUNT=$((TCAP + 1))
# gloo spins socket I/O threads per co-located rank; across WORLD ranks these
# compound the fork-bomb (imprint 2026-07-24).
export GLOO_SOCKET_NTHREADS="${GLOO_SOCKET_NTHREADS:-1}"
export GLOO_NSOCKS_PERTHREAD="${GLOO_NSOCKS_PERTHREAD:-1}"

# ---- MPS daemon --------------------------------------------------------------
# One daemon shared by every rank. Without it, co-located ranks time-slice the
# GPU and the stack buys nothing (~half throughput). The control-socket path
# MUST stay short: >108 bytes silently kills the daemon.
MPS_STARTED=0
if (( STACK > 1 )); then
  export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/m1p}"
  export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/m1p_log}"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  _DEVSTAMP="$CUDA_MPS_PIPE_DIRECTORY/.sonic_devices"
  # A leftover control socket from a KILLED prior run makes a naive `-S` check
  # skip startup even though no daemon is alive -> ranks silently time-slice and
  # the benchmark is invalid. Decide on daemon LIVENESS, not socket presence.
  if [[ -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]] && ! pgrep -u "$(id -u)" -f "nvidia-cuda-[m]ps-control" >/dev/null 2>&1; then
    echo "[sonic_mps] stale MPS control socket, no live daemon -> clearing $CUDA_MPS_PIPE_DIRECTORY"
    rm -f "$CUDA_MPS_PIPE_DIRECTORY"/control* "$_DEVSTAMP" 2>/dev/null
  fi
  if pgrep -u "$(id -u)" -f "nvidia-cuda-[m]ps-control" >/dev/null 2>&1; then
    # A live daemon's device set is not queryable, so we stamp it at start. A
    # daemon covering a DIFFERENT set silently renumbers our ranks onto the
    # wrong GPUs -- refuse rather than guess.
    _PREV="$(cat "$_DEVSTAMP" 2>/dev/null || echo '<unknown>')"
    if [[ "$_PREV" != "$MPS_DEVICES" ]]; then
      if [[ "${SONIC_MPS_FORCE_RESTART:-0}" == "1" ]]; then
        echo "[sonic_mps] live MPS daemon covers '$_PREV' != '$MPS_DEVICES' -> restarting (forced)"
        echo quit | nvidia-cuda-mps-control 2>/dev/null
        rm -f "$_DEVSTAMP"
      else
        echo "[sonic_mps] ERROR: a live MPS daemon covers devices '$_PREV' but this" >&2
        echo "[sonic_mps]        launch needs '$MPS_DEVICES'. Client device ids are" >&2
        echo "[sonic_mps]        relative to the SERVER's set, so reusing it would put" >&2
        echo "[sonic_mps]        ranks on the wrong GPUs. Re-run with" >&2
        echo "[sonic_mps]        SONIC_MPS_FORCE_RESTART=1 (kills the other daemon) or" >&2
        echo "[sonic_mps]        point CUDA_MPS_PIPE_DIRECTORY somewhere else." >&2
        exit 5
      fi
    else
      echo "[sonic_mps] reusing live MPS daemon pipe=$CUDA_MPS_PIPE_DIRECTORY devices=$MPS_DEVICES"
    fi
  fi
  if ! pgrep -u "$(id -u)" -f "nvidia-cuda-[m]ps-control" >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="$MPS_DEVICES" nvidia-cuda-mps-control -d && MPS_STARTED=1
    echo "$MPS_DEVICES" > "$_DEVSTAMP"
    echo "[sonic_mps] started MPS daemon pipe=$CUDA_MPS_PIPE_DIRECTORY devices=$MPS_DEVICES"
  fi
  cleanup_mps() {
    if (( MPS_STARTED == 1 )); then
      echo quit | nvidia-cuda-mps-control 2>/dev/null
      rm -f "$_DEVSTAMP" 2>/dev/null
    fi
  }
  trap cleanup_mps EXIT
  # ranks address the server's devices by RELATIVE index
  export CUDA_VISIBLE_DEVICES="$RANK_DEVICES"
else
  export CUDA_VISIBLE_DEVICES="$MPS_DEVICES"
fi

# ---- banner ------------------------------------------------------------------
echo "=== [sonic_mps] shape ==="
echo "[sonic_mps] NGPU=$NGPU STACK=$STACK -> world_size=$WORLD (requested; the run"
echo "[sonic_mps]   ABORTS in mps_stack.assert_world_size_matches_request if the"
echo "[sonic_mps]   realised accelerate world size is anything else)"
echo "[sonic_mps] num_envs=$NUM_ENVS/rank -> $((NUM_ENVS * WORLD)) envs total"
echo "[sonic_mps] backend=$( ((STACK > 1)) && echo gloo || echo nccl ) (NCCL rejects co-located ranks: 'Duplicate GPU detected')"
echo "[sonic_mps] threads/rank: omp=$TCAP tbb=$((TCAP + 1)) (nproc=$NPROC) stagger=${SONIC_RANK_STAGGER_SEC}s/rank"
echo "[sonic_mps] alloc_conf=$PYTORCH_CUDA_ALLOC_CONF ulimit-u=$(ulimit -u) ulimit-n=$(ulimit -n)"
echo "[sonic_mps] devices: physical='$MPS_DEVICES' -> ranks see CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"
echo "[sonic_mps] plant: arm KP=$PM_ARM_KP KD=$PM_ARM_KD effort shoulder(+yaw)=$PM_ARM_EFFORT_SHOULDER elbow=$PM_ARM_EFFORT_ELBOW wrist=$PM_ARM_EFFORT_WRIST"
echo "[sonic_mps] plant: enforced=$SONIC_REQUIRE_PRODUCTION_PLANT (proof -> <experiment_dir>/plant.yaml; env vars are NOT proof)"
echo "=== corpus provenance (must NOT resolve into /oscar/scratch) ==="
readlink -f "$MOTION"
stat -c '%n %s bytes' "$MOTION"
echo "=== gpus ==="
nvidia-smi -L
echo "=== node mem ==="
free -g | head -2

cd "$HERE"
"$ENVD/bin/accelerate" launch \
    --multi_gpu \
    --num_processes="$WORLD" \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port="$MAIN_PORT" \
    gear_sonic/train_agent_trl.py \
    +exp="$EXP" \
    exp_var="$EXP_VAR" \
    num_envs="$NUM_ENVS" headless=True \
    use_wandb=false \
    ${EXTRA_TRAIN_ARGS:-} \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file="$MOTION"
rc=$?
echo "[sonic_mps] EXIT=$rc"
exit $rc
