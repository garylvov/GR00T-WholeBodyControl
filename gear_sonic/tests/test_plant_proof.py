"""CPU-only guard tests for the H1-2 plant proof (``gear_sonic.utils.plant_proof``).

The defect this exists for: SONIC's arm group is env-gated, and the *unconfigured*
fallback (KP 19.74 / KD 1.26 / effort 40-18-18-19) is roughly half our production
stiffness and a third-to-a-sixth of our torque.  A run can silently train that
weaker robot while every log line and env var looks fine, so the check has to
read the gains back out of the simulation.

Run:  pytest gear_sonic/tests/test_plant_proof.py
"""

import types

import pytest
import yaml

from gear_sonic.utils import plant_proof


def _production_joint(group):
    return dict(plant_proof.H1_2_PRODUCTION_PLANT[group])


def _realised_production():
    """A full 27-DOF realised plant that IS the production one."""
    out = {}
    for side in ("left", "right"):
        for joint, group in (
            ("hip_yaw", "hip"),
            ("hip_roll", "hip"),
            ("hip_pitch", "hip"),
            ("knee", "knee"),
            ("ankle_pitch", "ankle_pitch"),
            ("ankle_roll", "ankle_roll"),
            ("shoulder_pitch", "shoulder"),
            ("shoulder_roll", "shoulder"),
            ("shoulder_yaw", "shoulder"),
            ("elbow", "elbow"),
            ("wrist_roll", "wrist"),
            ("wrist_pitch", "wrist"),
            ("wrist_yaw", "wrist"),
        ):
            spec = _production_joint(group)
            out[f"{side}_{joint}_joint"] = {
                k: spec[k] for k in ("stiffness", "damping", "effort_limit", "armature")
            }
    spec = _production_joint("torso")
    out["torso_joint"] = {k: spec[k] for k in ("stiffness", "damping", "effort_limit", "armature")}
    return out


def test_expectation_covers_all_27_h1_2_dofs():
    realised = _realised_production()
    assert len(realised) == 27  # H1-2 DOF count; the G1 wrist-index bug lives here
    assert plant_proof.check_expected_plant(realised) == []


def test_production_arm_values_are_the_teacher_ladder():
    """PM_ARM_KP=40 PM_ARM_KD=2.0 SHOULDER/ELBOW=120 WRIST=30."""
    plant = plant_proof.H1_2_PRODUCTION_PLANT
    for group in ("shoulder", "elbow", "wrist"):
        assert plant[group]["stiffness"] == 40.0
        assert plant[group]["damping"] == 2.0
    assert plant["shoulder"]["effort_limit"] == 120.0  # shoulder yaw shares this gate
    assert plant["elbow"]["effort_limit"] == 120.0
    assert plant["wrist"]["effort_limit"] == 30.0


def test_leg_groups_follow_the_armature_formula():
    """Legs/feet/torso are formula-derived and must match the teacher exactly."""
    plant = plant_proof.H1_2_PRODUCTION_PLANT
    nf, dr = plant_proof._NF, plant_proof._DR
    for group, armature in (("hip", 0.030), ("knee", 0.040), ("torso", 0.030)):
        assert plant[group]["stiffness"] == pytest.approx(armature * nf**2)
        assert plant[group]["damping"] == pytest.approx(2 * dr * armature * nf)
        assert plant[group]["armature"] == armature
    # ankles are the doubled rows
    assert plant["ankle_pitch"]["armature"] == pytest.approx(2 * 0.010)
    assert plant["ankle_roll"]["armature"] == pytest.approx(2 * 0.005)


def test_known_discrepancy_is_real_and_recorded():
    """KP 40 does NOT satisfy armature*w^2 (=19.74) at the declared armature.
    This test exists so the inconsistency stays visible instead of being
    'reconciled' by someone who assumes the formula holds everywhere."""
    plant = plant_proof.H1_2_PRODUCTION_PLANT
    armature = plant["shoulder"]["armature"]
    formula_kp = armature * plant_proof._NF**2
    assert formula_kp == pytest.approx(19.739, abs=0.01)
    assert plant["shoulder"]["stiffness"] == 40.0  # deliberately not formula_kp
    zeta = plant["shoulder"]["damping"] / (2 * armature * plant_proof._NF)
    assert zeta == pytest.approx(3.18, abs=0.02)


@pytest.mark.parametrize(
    "joint,field,wrong,label",
    [
        ("left_shoulder_pitch_joint", "stiffness", 19.739, "unconfigured KP"),
        ("left_shoulder_pitch_joint", "damping", 1.2566, "unconfigured KD"),
        ("left_shoulder_pitch_joint", "effort_limit", 40.0, "unconfigured shoulder effort"),
        ("left_shoulder_yaw_joint", "effort_limit", 18.0, "unconfigured shoulder-yaw effort"),
        ("left_elbow_joint", "effort_limit", 18.0, "unconfigured elbow effort"),
        ("left_wrist_roll_joint", "effort_limit", 19.0, "unconfigured wrist effort"),
        ("left_knee_joint", "stiffness", 100.0, "wrong knee stiffness"),
    ],
)
def test_the_unconfigured_fallback_is_caught(joint, field, wrong, label):
    realised = _realised_production()
    realised[joint][field] = wrong
    problems = plant_proof.check_expected_plant(realised)
    assert len(problems) == 1, label
    assert joint in problems[0] and field in problems[0]


def test_unknown_joint_is_reported_not_ignored():
    realised = _realised_production()
    realised["mystery_joint"] = {
        "stiffness": 1.0,
        "damping": 1.0,
        "effort_limit": 1.0,
        "armature": 1.0,
    }
    problems = plant_proof.check_expected_plant(realised)
    assert any("mystery_joint" in p for p in problems)


def _fake_articulation(realised):
    names = list(realised)

    class _T:
        def __init__(self, values):
            self._values = values

        def __getitem__(self, _idx):
            return types.SimpleNamespace(tolist=lambda: self._values)

    data = types.SimpleNamespace(
        joint_stiffness=_T([realised[n]["stiffness"] for n in names]),
        joint_damping=_T([realised[n]["damping"] for n in names]),
        joint_armature=_T([realised[n]["armature"] for n in names]),
        joint_effort_limits=_T([realised[n]["effort_limit"] for n in names]),
    )
    return types.SimpleNamespace(joint_names=names, data=data)


def test_dump_writes_a_readable_proof(tmp_path):
    realised = _realised_production()
    art = _fake_articulation(realised)
    out = tmp_path / "plant.yaml"
    report = plant_proof.dump_plant(art, out)
    assert report["mismatches_vs_production"] == []

    on_disk = yaml.safe_load(out.read_text())
    assert on_disk["realised"]["left_elbow_joint"]["effort_limit"] == 120.0
    assert on_disk["realised"]["left_shoulder_pitch_joint"]["stiffness"] == 40.0
    assert on_disk["realised"]["left_knee_joint"]["armature"] == 0.040
    assert on_disk["mismatches_vs_production"] == []


def test_dump_records_the_mismatch_rather_than_hiding_it(tmp_path):
    realised = _realised_production()
    realised["left_elbow_joint"]["effort_limit"] = 18.0
    out = tmp_path / "plant.yaml"
    report = plant_proof.dump_plant(_fake_articulation(realised), out)
    assert len(report["mismatches_vs_production"]) == 1
    assert "left_elbow_joint" in yaml.safe_load(out.read_text())["mismatches_vs_production"][0]
