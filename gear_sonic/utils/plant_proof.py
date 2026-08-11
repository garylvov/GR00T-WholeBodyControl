"""Write the *realised* actuator plant of a run to disk, per joint.

An exported env var proves what was requested; a log line proves what some code
said.  Neither proves what PhysX is actually simulating -- this campaign has been
bitten repeatedly by settings that were "set" and inert.  So every run dumps the
gains **read back out of the instantiated articulation** (``ArticulationData``,
i.e. the values written into the simulation) next to its config, and a run whose
plant is wrong is falsifiable from disk without re-reading any Python.

For H1-2 the production plant is our ProtoMotions teacher's
(``protomotions/robot_configs/h1_2.py`` under the canonical stiffen ladder
``PM_ARM_KP=40 PM_ARM_KD=2.0 PM_ARM_EFFORT_{SHOULDER,ELBOW}=120
PM_ARM_EFFORT_WRIST=30``); :data:`H1_2_PRODUCTION_PLANT` is that expectation and
:func:`check_expected_plant` is what turns a mismatch into a failure.
"""

from pathlib import Path

import yaml

# joint-name regex -> (stiffness, damping, effort_limit, armature)
# Values are the teacher's, from ProtoMotions robot_configs/h1_2.py with
# NATURAL_FREQ = 10*2*pi, DAMPING_RATIO = 2.0:
#   STIFFNESS_x = ARMATURE_x * NATURAL_FREQ**2
#   DAMPING_x   = 2 * DAMPING_RATIO * ARMATURE_x * NATURAL_FREQ
# The arm row is the production stiffen ladder, which deliberately does NOT
# satisfy those formulas -- see the KNOWN DISCREPANCY note in
# gear_sonic/envs/manager_env/robots/h1_2.py.
_NF = 10 * 2.0 * 3.1415926535
_DR = 2.0


def _pd(armature):
    return armature * _NF**2, 2.0 * _DR * armature * _NF


_K200, _D200 = _pd(0.030)
_K300, _D300 = _pd(0.040)
_K60, _D60 = _pd(0.010)
_K40, _D40 = _pd(0.005)

H1_2_PRODUCTION_PLANT = {
    "hip": {
        "match": ["_hip_yaw_joint", "_hip_roll_joint", "_hip_pitch_joint"],
        "stiffness": _K200,
        "damping": _D200,
        "effort_limit": 200.0,
        "armature": 0.030,
    },
    "knee": {
        "match": ["_knee_joint"],
        "stiffness": _K300,
        "damping": _D300,
        "effort_limit": 300.0,
        "armature": 0.040,
    },
    "ankle_pitch": {
        "match": ["_ankle_pitch_joint"],
        "stiffness": 2 * _K60,
        "damping": 2 * _D60,
        "effort_limit": 60.0,
        "armature": 2 * 0.010,
    },
    "ankle_roll": {
        "match": ["_ankle_roll_joint"],
        "stiffness": 2 * _K40,
        "damping": 2 * _D40,
        "effort_limit": 40.0,
        "armature": 2 * 0.005,
    },
    "torso": {
        "match": ["torso_joint"],
        "stiffness": _K200,
        "damping": _D200,
        "effort_limit": 200.0,
        "armature": 0.030,
    },
    # --- production stiffen ladder (NOT the unconfigured 19.74/1.26/40/18/19) --
    "shoulder": {
        "match": ["_shoulder_pitch_joint", "_shoulder_roll_joint", "_shoulder_yaw_joint"],
        "stiffness": 40.0,
        "damping": 2.0,
        "effort_limit": 120.0,
        "armature": 0.005,
    },
    "elbow": {
        "match": ["_elbow_joint"],
        "stiffness": 40.0,
        "damping": 2.0,
        "effort_limit": 120.0,
        "armature": 0.005,
    },
    "wrist": {
        "match": ["_wrist_roll_joint", "_wrist_pitch_joint", "_wrist_yaw_joint"],
        "stiffness": 40.0,
        "damping": 2.0,
        "effort_limit": 30.0,
        "armature": 0.005,
    },
}


def _group_of(joint_name: str, plant: dict):
    for group, spec in plant.items():
        if any(joint_name.endswith(m) or joint_name == m for m in spec["match"]):
            return group
    return None


def read_plant(articulation) -> dict:
    """Per-joint gains as PhysX has them, straight off ``ArticulationData``."""
    data = articulation.data
    names = list(articulation.joint_names)

    def row(tensor):
        return [float(v) for v in tensor[0].tolist()] if tensor is not None else [None] * len(names)

    stiffness = row(getattr(data, "joint_stiffness", None))
    damping = row(getattr(data, "joint_damping", None))
    armature = row(getattr(data, "joint_armature", None))
    effort = row(getattr(data, "joint_effort_limits", None))
    return {
        name: {
            "stiffness": stiffness[i],
            "damping": damping[i],
            "armature": armature[i],
            "effort_limit": effort[i],
        }
        for i, name in enumerate(names)
    }


def check_expected_plant(realised: dict, plant: dict = None, rtol: float = 1e-3) -> list:
    """Return a list of human-readable mismatch strings (empty == plant is right)."""
    plant = H1_2_PRODUCTION_PLANT if plant is None else plant
    problems = []
    for name, got in sorted(realised.items()):
        group = _group_of(name, plant)
        if group is None:
            problems.append(f"{name}: no expected group matches this joint")
            continue
        for field in ("stiffness", "damping", "effort_limit", "armature"):
            want = plant[group][field]
            have = got.get(field)
            if have is None:
                problems.append(f"{name}.{field}: simulation reported nothing")
            elif abs(have - want) > rtol * max(abs(want), 1.0):
                problems.append(f"{name}.{field}: realised {have:.6g}, production {want:.6g}")
    return problems


def dump_plant(articulation, path, plant: dict = None) -> dict:
    """Write the realised plant (+ any mismatches) to ``path``; return the report."""
    realised = read_plant(articulation)
    report = {
        "realised": realised,
        "mismatches_vs_production": check_expected_plant(realised, plant),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(report, fh, sort_keys=True)
    return report
