#!/usr/bin/env python3
# ADDED BY imprint (glvov) -- H1-2 embodiment for SONIC. Not upstream code.
"""Guard tests for the H1-2 embodiment: index round-trip + cross-model FK.

Upstream's docs say a wrong index table produces **no runtime error** -- the
robot just moves wrong.  So the tables and the assets are checked numerically:

T1  round-trip     : IsaacLab order -> MuJoCo order -> IsaacLab order == identity,
                     for DOF vectors, full qpos, and per-body tensors, using the
                     shipped tables and upstream's own ``convert()``.
T2  permutations   : both pairs are exact mutual inverses and are consistent with
                     the shipped URDF/MJCF (re-derived, not trusted).
T3  asset delta    : MuJoCo FK on the SONIC MJCF equals MuJoCo FK on our
                     canonical ProtoMotions ``h1_2_box_feet.xml`` for every
                     shared body, i.e. dropping ``head_aux`` and the visual mesh
                     geoms changed no kinematics.
T4  URDF == MJCF   : an independent FK walk of the shipped URDF equals MuJoCo FK
                     on the shipped MJCF, body by body.  This is where the
                     wrist-frame risk lives, so wrist links are reported
                     individually.
T5  end-to-end     : a pose authored in IsaacLab order, pushed through
                     ISAACLAB_TO_MUJOCO_DOF into MuJoCo, FK'd, and pulled back
                     through MUJOCO_TO_ISAACLAB_BODY, equals URDF FK evaluated
                     directly in IsaacLab order.  This is the test that actually
                     catches a scrambled permutation.

Run:  python gear_sonic/tests/test_h1_2_embodiment.py
Needs numpy + mujoco.  torch and isaaclab are optional (see _load_tables).
"""

from __future__ import annotations

import ast
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
URDF = REPO_ROOT / "gear_sonic/data/assets/robot_description/urdf/h1_2/h1_2.urdf"
MJCF = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/h1_2.xml"
ROBOT_PY = REPO_ROOT / "gear_sonic/envs/manager_env/robots/h1_2.py"
CANONICAL_MJCF = Path(
    "/oscar/scratch/glvov/imprint-retread-bump/third_party/ProtoMotions/"
    "protomotions/data/assets/mjcf/h1_2_box_feet.xml"
)

POS_TOL = 1e-9   # metres; these are the *same numbers* re-expressed, not a fit
ROT_TOL = 1e-9   # Frobenius distance between rotation matrices

_FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("[ok]   " if cond else "[FAIL] ") + msg)
    if not cond:
        _FAILURES.append(msg)


# --------------------------------------------------------------------------- #
def _load_tables() -> dict:
    """Import the real module; fall back to parsing its literals.

    ``robots/h1_2.py`` imports isaaclab at module scope, which is unavailable
    outside the groot-sonic env.  The tables are plain literals, so parsing them
    tests exactly the same shipped values.
    """
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from gear_sonic.envs.manager_env.robots import h1_2 as mod  # noqa: PLC0415

        print("[info] loaded robots/h1_2.py natively (isaaclab present)")
        return {
            n: getattr(mod, n)
            for n in (
                "H1_2_ISAACLAB_JOINTS",
                "H1_2_ISAACLAB_TO_MUJOCO_DOF",
                "H1_2_MUJOCO_TO_ISAACLAB_DOF",
                "H1_2_ISAACLAB_TO_MUJOCO_BODY",
                "H1_2_MUJOCO_TO_ISAACLAB_BODY",
            )
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[info] native import unavailable ({type(exc).__name__}); parsing literals")
        src = ROBOT_PY.read_text()
        out = {}
        for n in (
            "H1_2_ISAACLAB_JOINTS",
            "H1_2_ISAACLAB_TO_MUJOCO_DOF",
            "H1_2_MUJOCO_TO_ISAACLAB_DOF",
            "H1_2_ISAACLAB_TO_MUJOCO_BODY",
            "H1_2_MUJOCO_TO_ISAACLAB_BODY",
        ):
            m = re.search(re.escape(n) + r"\s*=\s*(\[[^\]]*\])", src)
            out[n] = ast.literal_eval(m.group(1))
        return out


# --------------------------------------------------------------------------- #
# minimal URDF forward kinematics (independent of MuJoCo)
# --------------------------------------------------------------------------- #
def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = (
        math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def axis_angle_to_mat(axis, angle):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


class UrdfModel:
    def __init__(self, path: Path):
        root = ET.parse(path).getroot()
        self.links = [ln.get("name") for ln in root.findall("link")]
        self.joints = []
        self.children = {ln: [] for ln in self.links}
        child_links = set()
        for j in root.findall("joint"):
            if j.get("type") == "fixed":
                continue
            org = j.find("origin")
            xyz = [float(v) for v in (org.get("xyz", "0 0 0")).split()]
            rpy = [float(v) for v in (org.get("rpy", "0 0 0")).split()]
            rec = {
                "name": j.get("name"),
                "parent": j.find("parent").get("link"),
                "child": j.find("child").get("link"),
                "xyz": np.array(xyz),
                "R": rpy_to_mat(*rpy),
                "axis": [float(v) for v in j.find("axis").get("xyz").split()],
            }
            self.joints.append(rec)
            self.children[rec["parent"]].append(rec)
            child_links.add(rec["child"])
        self.root = next(ln for ln in self.links if ln not in child_links)
        # BFS with alphabetical sibling sort == Isaac Lab's URDF importer order
        self.body_order = [self.root]
        self.joint_order = []
        queue = [self.root]
        while queue:
            nxt = []
            for parent in queue:
                for rec in sorted(self.children[parent], key=lambda r: r["child"]):
                    self.body_order.append(rec["child"])
                    self.joint_order.append(rec["name"])
                    nxt.append(rec["child"])
            queue = nxt
        self._by_name = {j["name"]: j for j in self.joints}

    def fk(self, q_by_joint_name: dict) -> dict:
        """Return {link: (3x3 R, 3 t)} in the world frame, root at identity."""
        out = {self.root: (np.eye(3), np.zeros(3))}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            Rp, tp = out[parent]
            for rec in self.children[parent]:
                Rj = rec["R"] @ axis_angle_to_mat(rec["axis"], q_by_joint_name[rec["name"]])
                out[rec["child"]] = (Rp @ Rj, tp + Rp @ rec["xyz"])
                stack.append(rec["child"])
        return out


# --------------------------------------------------------------------------- #
def mujoco_fk(path: Path, q_by_joint_name: dict):
    import mujoco  # noqa: PLC0415

    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0  # free-joint quat w
    for name, val in q_by_joint_name.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = val
    mujoco.mj_kinematics(m, d)
    out = {}
    for i in range(m.nbody):
        name = m.body(i).name
        if not name or name == "world":
            continue
        out[name] = (d.xmat[i].reshape(3, 3).copy(), d.xpos[i].copy())
    return m, out


def rot_err(Ra, Rb) -> float:
    """Frobenius distance between rotation matrices.

    Deliberately NOT the geodesic acos((tr-1)/2): near R=I that formula has a
    sqrt(eps) ~ 1.5e-8 numerical floor in float64 and would report a spurious
    "error" for matrices that are bit-identical.  Frobenius is linear in the
    perturbation and bottoms out at true machine epsilon.
    """
    return float(np.linalg.norm(Ra - Rb))


def compare(a: dict, b: dict, names, label: str, verbose_names=()):
    dp = dr = 0.0
    worst = ""
    for n in names:
        Ra, ta = a[n]
        Rb, tb = b[n]
        # both FKs put the root at the identity; compare relative to root
        p = float(np.linalg.norm(ta - tb))
        r = rot_err(Ra, Rb)
        if max(p, r) > max(dp, dr):
            worst = n
        dp, dr = max(dp, p), max(dr, r)
        if n in verbose_names:
            print(f"        {n:<26s} |dp|={p:.3e} m  |dR|={r:.3e}")
    print(f"       max |dp| = {dp:.3e} m, max |dR| = {dr:.3e} (worst body: {worst})")
    check(dp < POS_TOL and dr < ROT_TOL, f"{label} (tol {POS_TOL:g} m / {ROT_TOL:g} Frob)")


# --------------------------------------------------------------------------- #

def _t7(il_bodies: list, n_dof: int) -> None:
    """Every hardcoded joint index the H1-2 experiment uses must be in range.

    The shared observation term
    ``config/manager_env/observations/terms/joint_pos_multi_future_wrist_for_smpl.yaml``
    hardcodes G1's six wrist DOF indices ``[23, 24, 25, 26, 27, 28]``.  G1 has
    29 DOF so those are its trailing six; H1-2 has 27, so 27 and 28 are off the
    end.  Nothing type-checks this -- it surfaced only as a CUDA device-side
    ``index out of bounds`` assert thrown from observations.py during
    ObservationManager construction, which is both late and unreadable.

    So: parse the indices the H1-2 experiment actually composes, and check them
    against the embodiment tables re-derived in main().  Also assert they name
    the six wrist DOFs, so a merely in-range but wrong list still fails.
    """
    import yaml  # noqa: PLC0415

    exp = REPO_ROOT / (
        "gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h1_2.yaml"
    )
    shared = REPO_ROOT / (
        "gear_sonic/config/manager_env/observations/terms/"
        "joint_pos_multi_future_wrist_for_smpl.yaml"
    )
    cfg = yaml.safe_load(exp.read_text())
    idx = (
        cfg.get("manager_env", {})
        .get("observations", {})
        .get("tokenizer", {})
        .get("joint_pos_multi_future_wrist_for_smpl", {})
        .get("params", {})
        .get("joints_idx")
    )
    check(
        idx is not None,
        "sonic_h1_2.yaml overrides joint_pos_multi_future_wrist_for_smpl.joints_idx "
        "(the shared G1 default is out of bounds on H1-2)",
    )
    if idx is None:
        return

    check(
        all(0 <= i < n_dof for i in idx),
        f"all H1-2 wrist joints_idx in [0, {n_dof}): {idx}",
    )

    # DOF index = body index - 1; entry 0 of H1_2_ISAACLAB_JOINTS is the root.
    expected = [il_bodies.index(n) - 1 for n in (
        "left_wrist_roll_link", "right_wrist_roll_link",
        "left_wrist_pitch_link", "right_wrist_pitch_link",
        "left_wrist_yaw_link", "right_wrist_yaw_link",
    )]
    check(
        list(idx) == expected,
        f"H1-2 wrist joints_idx names the six wrist DOFs in the shared term's "
        f"order (expected {expected}, got {list(idx)})",
    )

    # Guard the premise: the shared default really is G1's and really is unusable
    # here, so this override is not cargo-culted and must not be quietly dropped.
    g1_idx = yaml.safe_load(shared.read_text())[
        "joint_pos_multi_future_wrist_for_smpl"
    ]["params"]["joints_idx"]
    check(
        any(i >= n_dof for i in g1_idx),
        f"shared G1 default {g1_idx} is out of range for H1-2's {n_dof} DOF, "
        "which is why the override exists",
    )


def _t6() -> None:
    """SONIC's Humanoid_Batch must yield a 27-wide dof_pos for H1-2.

    This is the head_aux question, settled numerically on upstream's own code
    rather than by argument.  fk_batch derives dof_pos two ways:

        extend_config non-empty : pose[..., 1 : num_bodies]        (naive slice)
        extend_config empty     : pose[..., actuated_joints_idx]   (if that list
                                  is shorter than body_names) else pose[..., 1:]

    Both assume every non-root body owns exactly one DOF.  Our canonical
    ProtoMotions asset has 29 bodies / 27 DOF because ``head_aux`` is a massless
    marker with no joint, so BOTH branches return 28 -- and the actuated_joints_idx
    branch is worse than off-by-one, because body_to_joint also picks up pelvis
    (it owns the free joint), so slot 0 is the root. Every DOF would be shifted.

    Dropping head_aux from the SONIC asset (28 bodies / 27 DOF) makes both
    branches correct, and only then does extend_config become usable: it restores
    head_aux as an augmented body (num_bodies_augment 29) while dof_pos stays 27.
    """
    try:
        import types  # noqa: PLC0415

        for _m in ("open3d", "open3d.geometry", "open3d.utility", "open3d.io"):
            if _m not in sys.modules:
                sys.modules[_m] = types.ModuleType(_m)
        sys.modules["open3d"].io = sys.modules["open3d.io"]
        sys.modules["open3d"].geometry = sys.modules["open3d.geometry"]
        sys.modules["open3d"].utility = sys.modules["open3d.utility"]
        sys.modules["open3d.io"].read_triangle_mesh = lambda *a, **k: types.SimpleNamespace(
            vertices=[], triangles=[]
        )
        from easydict import EasyDict  # noqa: PLC0415

        sys.path.insert(0, str(REPO_ROOT))
        from gear_sonic.utils.motion_lib.torch_humanoid_batch import (  # noqa: PLC0415
            Humanoid_Batch,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIP] Humanoid_Batch unavailable ({type(exc).__name__}: {exc})")
        _FAILURES.append("T6 skipped: Humanoid_Batch deps unavailable")
        return

    head = [
        EasyDict(
            {
                "joint_name": "head_aux",
                "parent_name": "torso_link",
                "pos": [0.0, 0.0, 0.7],
                "rot": [1.0, 0.0, 0.0, 0.0],
            }
        )
    ]

    def dof_width(root: str, fname: str, extend):
        h = Humanoid_Batch(EasyDict({"asset": {"assetRoot": root, "assetFileName": fname},
                                     "extend_config": extend}))
        if len(extend) > 0:
            w = h.num_bodies - 1
        elif len(h.actuated_joints_idx) != len(h.body_names):
            w = len(h.actuated_joints_idx)
        else:
            w = h.num_bodies - 1
        return h, w

    sonic_root = str(REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf") + "/"
    for extend, tag in [([], "extend_config=[]"), (head, "extend_config=[head_aux]")]:
        h, w = dof_width(sonic_root, "h1_2.xml", extend)
        print(f"   shipped 28-body asset, {tag}: num_bodies={h.num_bodies} "
              f"augment={h.num_bodies_augment} dof_pos_width={w}")
        check(w == 27, f"shipped asset with {tag} yields 27-wide dof_pos")
    if CANONICAL_MJCF.exists():
        canon_root = str(CANONICAL_MJCF.parent) + "/"
        for extend, tag in [([], "extend_config=[]"), (head, "extend_config=[head_aux]")]:
            h, w = dof_width(canon_root, CANONICAL_MJCF.name, extend)
            print(f"   29-body head_aux asset, {tag}: num_bodies={h.num_bodies} "
                  f"augment={h.num_bodies_augment} dof_pos_width={w}")
            check(w == 28, f"29-body asset with {tag} yields 28 (documents WHY head_aux is dropped)")


def main() -> int:
    rng = np.random.default_rng(20260806)
    tables = _load_tables()
    il_bodies = tables["H1_2_ISAACLAB_JOINTS"]
    i2m_dof = tables["H1_2_ISAACLAB_TO_MUJOCO_DOF"]
    m2i_dof = tables["H1_2_MUJOCO_TO_ISAACLAB_DOF"]
    i2m_body = tables["H1_2_ISAACLAB_TO_MUJOCO_BODY"]
    m2i_body = tables["H1_2_MUJOCO_TO_ISAACLAB_BODY"]

    urdf = UrdfModel(URDF)
    il_joints = urdf.joint_order
    n_dof = len(il_joints)

    print("\n=== T2: permutation sanity ===")
    check(len(il_bodies) == n_dof + 1 == 28, f"28 bodies / 27 DOF (got {len(il_bodies)}/{n_dof})")
    check(il_bodies == urdf.body_order, "H1_2_ISAACLAB_JOINTS == URDF BFS-alphabetical link order")
    check(
        all(m2i_dof[v] == k for k, v in enumerate(i2m_dof)),
        "DOF tables are exact mutual inverses",
    )
    check(
        all(m2i_body[v] == k for k, v in enumerate(i2m_body)),
        "BODY tables are exact mutual inverses",
    )
    # re-derive from the assets rather than trusting the literals
    m_model, _ = mujoco_fk(MJCF, dict.fromkeys(il_joints, 0.0))
    mj_bodies = [m_model.body(i).name for i in range(m_model.nbody) if m_model.body(i).name != "world"]
    mj_joints = [
        m_model.joint(i).name for i in range(m_model.njnt) if m_model.joint(i).type[0] != 0
    ]
    check(
        i2m_body == [il_bodies.index(n) for n in mj_bodies]
        and m2i_body == [mj_bodies.index(n) for n in il_bodies],
        "BODY tables re-derived from the shipped MJCF/URDF match the frozen literals",
    )
    check(
        i2m_dof == [il_joints.index(n) for n in mj_joints]
        and m2i_dof == [mj_joints.index(n) for n in il_joints],
        "DOF tables re-derived from the shipped MJCF/URDF match the frozen literals",
    )

    print("\n=== T1: round-trip through upstream's converter ===")
    try:
        import torch  # noqa: PLC0415

        sys.path.insert(0, str(REPO_ROOT))
        from gear_sonic.trl.utils.order_converter import IsaacLabMuJoCoConverter  # noqa: PLC0415

        class _H1_2(IsaacLabMuJoCoConverter):
            JOINT_NAMES = il_bodies
            DOF_MAPPINGS = {("isaaclab", "mujoco"): i2m_dof, ("mujoco", "isaaclab"): m2i_dof}
            BODY_MAPPINGS = {("isaaclab", "mujoco"): i2m_body, ("mujoco", "isaaclab"): m2i_body}
            VR_3POINTS_BODY_NAMES = [
                "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
            ]
            FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

        conv = _H1_2()
        check(conv.num_dof == 27, f"converter.num_dof == 27 (got {conv.num_dof})")
        for label, shape in [
            ("dof [T, 27]", (11, n_dof)),
            ("qpos [T, 7+27]", (11, 7 + n_dof)),
            ("body xf [T, 28, 3]", (11, n_dof + 1, 3)),
            ("body rot [T, 28, 3, 3]", (5, n_dof + 1, 3, 3)),
        ]:
            x = torch.randn(*shape)
            err = (conv.to_isaaclab(conv.to_mujoco(x)) - x).abs().max().item()
            check(err == 0.0, f"round-trip exact for {label} (max abs err {err:g})")
        # qpos root block must be untouched
        x = torch.randn(4, 7 + n_dof)
        check(
            torch.equal(conv.to_mujoco(x)[:, :7], x[:, :7]),
            "qpos root block (trans+quat) passes through unpermuted",
        )
        # semantic check: named joint values land on the right MuJoCo slots
        vals = torch.arange(float(n_dof)).unsqueeze(0)
        mj_vals = conv.to_mujoco(vals)[0].tolist()
        check(
            all(mj_vals[k] == float(il_joints.index(n)) for k, n in enumerate(mj_joints)),
            "to_mujoco() places each named joint at its MuJoCo index",
        )
        print(f"       vr_3points_mujoco_indices = {conv.vr_3points_mujoco_indices}")
        print(f"       foot_mujoco_indices       = {conv.foot_mujoco_indices}")
        check(
            [mj_bodies[i] for i in conv.vr_3points_mujoco_indices]
            == ["torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"],
            "vr_3points_mujoco_indices resolve to the intended bodies",
        )
        check(
            [mj_bodies[i] for i in conv.foot_mujoco_indices]
            == ["left_ankle_roll_link", "right_ankle_roll_link"],
            "foot_mujoco_indices resolve to the intended bodies",
        )
    except ImportError as exc:
        print(f"[SKIP] torch unavailable ({exc}); round-trip via converter not run")
        _FAILURES.append("T1 skipped: torch unavailable")

    # ---------------- FK tests ---------------- #
    wrists = [n for n in il_bodies if "wrist" in n]
    poses = []
    for _ in range(6):
        poses.append({n: float(v) for n, v in zip(il_joints, rng.uniform(-0.4, 0.4, n_dof))})
    poses.append(dict.fromkeys(il_joints, 0.0))

    print("\n=== T3: SONIC MJCF vs our canonical ProtoMotions box_feet MJCF ===")
    if CANONICAL_MJCF.exists():
        for i, q in enumerate(poses):
            _, a = mujoco_fk(MJCF, q)
            _, b = mujoco_fk(CANONICAL_MJCF, q)
            shared = [n for n in a if n in b]
            print(f"   pose {i}: {len(shared)} shared bodies")
            compare(a, b, shared, f"pose {i}: SONIC MJCF == canonical box_feet MJCF",
                    verbose_names=wrists if i == 0 else ())
    else:
        print(f"[SKIP] canonical MJCF not found at {CANONICAL_MJCF}")
        _FAILURES.append("T3 skipped: canonical MJCF missing")

    print("\n=== T4: shipped URDF vs shipped MJCF (wrist frames called out) ===")
    for i, q in enumerate(poses):
        a = urdf.fk(q)
        _, b = mujoco_fk(MJCF, q)
        names = [n for n in urdf.links if n in b]
        print(f"   pose {i}:")
        compare(a, b, names, f"pose {i}: URDF FK == MJCF FK",
                verbose_names=wrists if i in (0, 1) else ())

    print("\n=== T5: end-to-end permutation (Isaac order -> MuJoCo -> back) ===")
    for i, q in enumerate(poses[:3]):
        q_isaac = np.array([q[n] for n in il_joints])
        q_mujoco = q_isaac[i2m_dof]                      # gather: isaac -> mujoco
        _, mj = mujoco_fk(MJCF, dict(zip(mj_joints, q_mujoco)))
        mj_pos = np.array([mj[n][1] for n in mj_bodies])
        back_pos = mj_pos[m2i_body]                      # gather: mujoco -> isaac
        ref = urdf.fk(q)
        ref_pos = np.array([ref[n][1] for n in il_bodies])
        d = float(np.abs(back_pos - ref_pos).max())
        print(f"   pose {i}: max |dp| = {d:.3e} m")
        check(d < POS_TOL, f"pose {i}: full Isaac->MuJoCo->Isaac body pipeline is identity")

    print("\n=== T6: Humanoid_Batch dof_pos width (head_aux / extend_config) ===")
    _t6()

    print("\n=== T7: hardcoded wrist joint indices are in range for H1-2 ===")
    _t7(il_bodies, n_dof)

    print("\n" + "=" * 72)
    if _FAILURES:
        print(f"FAILED ({len(_FAILURES)}):")
        for f in _FAILURES:
            print("  - " + f)
        return 1
    print("ALL H1-2 EMBODIMENT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
