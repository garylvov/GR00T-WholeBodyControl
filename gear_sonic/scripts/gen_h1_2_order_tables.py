#!/usr/bin/env python3
# ADDED BY imprint (glvov) -- H1-2 embodiment for SONIC. Not upstream code.
"""Derive (and self-verify) the IsaacLab<->MuJoCo index tables for H1-2.

Why a generator: upstream's own docs call the DOF-order mismatch the #1 trap and
note that a wrong permutation produces **no runtime error** -- the robot simply
moves wrong.  Hand-typing 4 x 27/28-element permutations is the classic silent
failure, so the tables in ``robots/h1_2.py`` are produced here and frozen as
literals.  Re-run this script after any asset change and diff the output.

ORDERING RULES USED
-------------------
MuJoCo    : DFS over ``<body>`` elements in XML order (excluding ``world``).
Isaac Lab : BFS over the URDF link tree, siblings sorted **alphabetically** by
            link name (upstream ``docs/source/references/conventions.md``,
            "Traversal order details").

TABLE SEMANTICS (reverse-engineered from upstream's own H2 tables and
re-verified by ``verify_h2_convention()`` below -- do not guess these)

    ISAACLAB_TO_MUJOCO_X[k] = index, in IsaacLab order, of the k-th MuJoCo entry
    MUJOCO_TO_ISAACLAB_X[i] = index, in MuJoCo order, of the i-th IsaacLab entry

i.e. both are *gather* indices: ``mujoco_ordered = isaac_ordered[ISAACLAB_TO_MUJOCO_X]``
and ``isaac_ordered = mujoco_ordered[MUJOCO_TO_ISAACLAB_X]``.  This matches
``IsaacLabMuJoCoConverter.convert`` (``data[..., mapping]``) and
``motion_lib_base`` (``body_pos_w[:, mujoco_to_isaaclab_body]``).

(Note: PR #112's ``x2_ultra.py`` uses the *opposite* naming for its module-level
constants and then swaps them when building the mapping dict.  We follow G1/H2
naming so the constant name states the truth.)

CROSS-CHECKS PERFORMED
----------------------
1. URDF tree and MJCF tree describe the same robot (names, parents, joints).
2. The two permutations are exact mutual inverses.
3. The derived IsaacLab joint order equals the order our own r13 USD factory
   froze as ``EXPECTED_NO_FINGERS_ISAAC_JOINT_ORDER`` (an *empirically measured*
   Isaac order in our stack) after applying the factory's documented arm-joint
   relabel.  Two independent sources agreeing is the real evidence here.
4. The same derivation rules, applied to upstream's ``h2.xml``, reproduce
   upstream's published H2 tables byte for byte.

Usage
-----
    python gear_sonic/scripts/gen_h1_2_order_tables.py            # print + verify
    python gear_sonic/scripts/gen_h1_2_order_tables.py --check    # verify only
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
URDF = REPO_ROOT / "gear_sonic/data/assets/robot_description/urdf/h1_2/h1_2.urdf"
MJCF = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/h1_2.xml"
H2_MJCF = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/h2.xml"
H2_PY = REPO_ROOT / "gear_sonic/envs/manager_env/robots/h2.py"

# --------------------------------------------------------------------------- #
# Frozen expectation from OUR r13 USD factory
# (imprint/integrations/unitree_lab/h1_2_usd.py ::
#  EXPECTED_NO_FINGERS_ISAAC_JOINT_ORDER).  That list is the Isaac Lab joint
# order measured at runtime in our own stack and asserted by the factory's
# validator, so it is an independent witness for the BFS-alphabetical rule.
# The factory relabels two forearm axes; box_feet (our canonical MJCF, and hence
# this embodiment) uses the other spelling -- see BOX_FEET_ARM_JOINT_RENAMES in
# the factory.
# --------------------------------------------------------------------------- #
FACTORY_TO_BOXFEET_JOINT = {
    "left_elbow_pitch_joint": "left_elbow_joint",
    "left_elbow_roll_joint": "left_wrist_roll_joint",
    "right_elbow_pitch_joint": "right_elbow_joint",
    "right_elbow_roll_joint": "right_wrist_roll_joint",
}
FACTORY_NO_FINGERS_ISAAC_JOINT_ORDER = [
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "torso_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_elbow_roll_joint",
    "right_elbow_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]


# --------------------------------------------------------------------------- #
# parsers
# --------------------------------------------------------------------------- #
def mujoco_order(mjcf: Path) -> tuple[list[str], list[str], dict[str, str]]:
    """DFS body order (root first) + DFS joint order + body->parent map."""
    root = ET.parse(mjcf).getroot()
    bodies: list[str] = []
    joints: list[str] = []
    parent_of: dict[str, str] = {}

    def walk(el, parent):
        for b in el.findall("body"):
            name = b.get("name")
            bodies.append(name)
            if parent is not None:
                parent_of[name] = parent
            for j in b.findall("joint"):
                if j.get("type") != "free":
                    joints.append(j.get("name"))
            walk(b, name)

    walk(root.find("worldbody"), None)
    return bodies, joints, parent_of


def isaaclab_order(urdf: Path) -> tuple[list[str], list[str], dict[str, str]]:
    """BFS link order with alphabetical sibling sort + matching joint order."""
    root = ET.parse(urdf).getroot()
    links = [ln.get("name") for ln in root.findall("link")]
    children: dict[str, list[tuple[str, str]]] = {ln: [] for ln in links}
    child_links = set()
    parent_of: dict[str, str] = {}
    for j in root.findall("joint"):
        if j.get("type") == "fixed":
            # Isaac Lab's URDF importer merges fixed joints; they carry no DOF.
            continue
        p = j.find("parent").get("link")
        c = j.find("child").get("link")
        children[p].append((c, j.get("name")))
        child_links.add(c)
        parent_of[c] = p
    roots = [ln for ln in links if ln not in child_links]
    if len(roots) != 1:
        raise SystemExit(f"expected exactly one root link in {urdf}, got {roots}")

    body_order = [roots[0]]
    joint_order: list[str] = []
    queue = [roots[0]]
    while queue:
        nxt = []
        for parent in queue:
            for child, joint in sorted(children[parent], key=lambda t: t[0]):
                body_order.append(child)
                joint_order.append(joint)
                nxt.append(child)
        queue = nxt
    return body_order, joint_order, parent_of


# --------------------------------------------------------------------------- #
# table construction
# --------------------------------------------------------------------------- #
def gather_tables(isaac: list[str], mujoco: list[str]) -> tuple[list[int], list[int]]:
    """Return (ISAACLAB_TO_MUJOCO, MUJOCO_TO_ISAACLAB) for one name list pair."""
    if sorted(isaac) != sorted(mujoco):
        only_i = sorted(set(isaac) - set(mujoco))
        only_m = sorted(set(mujoco) - set(isaac))
        raise SystemExit(f"name sets differ.\n  only in isaac: {only_i}\n  only in mujoco: {only_m}")
    i_index = {n: i for i, n in enumerate(isaac)}
    m_index = {n: i for i, n in enumerate(mujoco)}
    isaaclab_to_mujoco = [i_index[n] for n in mujoco]
    mujoco_to_isaaclab = [m_index[n] for n in isaac]
    return isaaclab_to_mujoco, mujoco_to_isaaclab


def assert_inverse(a: list[int], b: list[int], label: str) -> None:
    if len(a) != len(b):
        raise SystemExit(f"{label}: length mismatch {len(a)} vs {len(b)}")
    for k, v in enumerate(a):
        if b[v] != k:
            raise SystemExit(f"{label}: not mutual inverses at k={k} (a[{k}]={v}, b[{v}]={b[v]})")


# --------------------------------------------------------------------------- #
# convention guard: reproduce upstream's H2 tables from h2.xml + h2.py
# --------------------------------------------------------------------------- #
def _literal_list(src: str, name: str):
    m = re.search(re.escape(name) + r"\s*=\s*(\[[^\]]*\])", src)
    if not m:
        raise SystemExit(f"could not find {name} in {H2_PY}")
    import ast

    return ast.literal_eval(m.group(1))


def verify_h2_convention() -> None:
    """The exact same code path must reproduce upstream's own H2 constants."""
    if not (H2_MJCF.exists() and H2_PY.exists()):
        print("[warn] h2 assets missing; skipping convention guard", file=sys.stderr)
        return
    src = H2_PY.read_text()
    isaac_bodies = _literal_list(src, "H2_ISAACLAB_JOINTS")
    mj_bodies, mj_joints, _ = mujoco_order(H2_MJCF)

    i2m_body, m2i_body = gather_tables(isaac_bodies, mj_bodies)
    if i2m_body != _literal_list(src, "H2_ISAACLAB_TO_MUJOCO_BODY"):
        raise SystemExit("convention guard FAILED: H2 body isaaclab->mujoco mismatch")
    if m2i_body != _literal_list(src, "H2_MUJOCO_TO_ISAACLAB_BODY"):
        raise SystemExit("convention guard FAILED: H2 body mujoco->isaaclab mismatch")

    # DOF order follows body order minus the root, in both conventions.
    isaac_dof_bodies = isaac_bodies[1:]
    mj_dof_bodies = [b for b in mj_bodies[1:]]
    if len(mj_joints) != len(mj_dof_bodies):
        raise SystemExit("H2: one-joint-per-body assumption broken")
    i2m_dof, m2i_dof = gather_tables(isaac_dof_bodies, mj_dof_bodies)
    if i2m_dof != _literal_list(src, "H2_ISAACLAB_TO_MUJOCO_DOF"):
        raise SystemExit("convention guard FAILED: H2 dof isaaclab->mujoco mismatch")
    if m2i_dof != _literal_list(src, "H2_MUJOCO_TO_ISAACLAB_DOF"):
        raise SystemExit("convention guard FAILED: H2 dof mujoco->isaaclab mismatch")
    print("[ok] convention guard: reproduced upstream's four H2 tables exactly")


# --------------------------------------------------------------------------- #
def build():
    mj_bodies, mj_joints, mj_parent = mujoco_order(MJCF)
    il_bodies, il_joints, il_parent = isaaclab_order(URDF)

    # (1) same robot?
    if sorted(mj_bodies) != sorted(il_bodies):
        raise SystemExit("URDF and MJCF body name sets differ")
    if sorted(mj_joints) != sorted(il_joints):
        raise SystemExit("URDF and MJCF joint name sets differ")
    if mj_parent != il_parent:
        diff = {k: (mj_parent.get(k), il_parent.get(k)) for k in set(mj_parent) | set(il_parent)
                if mj_parent.get(k) != il_parent.get(k)}
        raise SystemExit(f"URDF and MJCF tree structures differ: {diff}")
    if len(mj_bodies) != len(mj_joints) + 1:
        raise SystemExit(
            f"num_bodies({len(mj_bodies)}) != num_dof+1({len(mj_joints) + 1}); "
            "SONIC's converter requires this invariant"
        )
    print(f"[ok] URDF and MJCF agree: {len(mj_bodies)} bodies, {len(mj_joints)} DOF")

    # (2) tables
    i2m_body, m2i_body = gather_tables(il_bodies, mj_bodies)
    i2m_dof, m2i_dof = gather_tables(il_joints, mj_joints)
    assert_inverse(i2m_body, m2i_body, "BODY")
    assert_inverse(i2m_dof, m2i_dof, "DOF")
    print("[ok] both permutation pairs are exact mutual inverses")

    # (3) agreement with our r13 factory's measured Isaac order
    expected = [FACTORY_TO_BOXFEET_JOINT.get(n, n) for n in FACTORY_NO_FINGERS_ISAAC_JOINT_ORDER]
    if il_joints != expected:
        raise SystemExit(
            "derived IsaacLab joint order disagrees with the r13 factory's\n"
            f"  derived : {il_joints}\n  factory : {expected}"
        )
    print("[ok] derived IsaacLab order == r13 factory EXPECTED_NO_FINGERS_ISAAC_JOINT_ORDER")

    return il_bodies, il_joints, mj_bodies, mj_joints, i2m_dof, m2i_dof, i2m_body, m2i_body


def emit(il_bodies, i2m_dof, m2i_dof, i2m_body, m2i_body) -> str:
    def block(name, vals, per_line=8):
        rows = [
            "    " + ", ".join(f"{v}" for v in vals[i : i + per_line]) + ","
            for i in range(0, len(vals), per_line)
        ]
        return f"{name} = [\n" + "\n".join(rows) + "\n]"

    lines = ["H1_2_ISAACLAB_JOINTS = ["]
    lines += [f'    "{n}",' for n in il_bodies]
    lines += ["]", ""]
    lines += [block("H1_2_ISAACLAB_TO_MUJOCO_DOF", i2m_dof), ""]
    lines += [block("H1_2_MUJOCO_TO_ISAACLAB_DOF", m2i_dof), ""]
    lines += [block("H1_2_ISAACLAB_TO_MUJOCO_BODY", i2m_body), ""]
    lines += [block("H1_2_MUJOCO_TO_ISAACLAB_BODY", m2i_body)]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only, do not print tables")
    args = ap.parse_args()

    verify_h2_convention()
    il_bodies, il_joints, mj_bodies, mj_joints, i2m_dof, m2i_dof, i2m_body, m2i_body = build()

    if args.check:
        # Compare against the frozen literals actually shipped.
        sys.path.insert(0, str(REPO_ROOT))
        src = (REPO_ROOT / "gear_sonic/envs/manager_env/robots/h1_2.py").read_text()
        import ast

        for name, want in [
            ("H1_2_ISAACLAB_JOINTS", il_bodies),
            ("H1_2_ISAACLAB_TO_MUJOCO_DOF", i2m_dof),
            ("H1_2_MUJOCO_TO_ISAACLAB_DOF", m2i_dof),
            ("H1_2_ISAACLAB_TO_MUJOCO_BODY", i2m_body),
            ("H1_2_MUJOCO_TO_ISAACLAB_BODY", m2i_body),
        ]:
            m = re.search(re.escape(name) + r"\s*=\s*(\[[^\]]*\])", src)
            if m is None or ast.literal_eval(m.group(1)) != want:
                raise SystemExit(f"FROZEN LITERAL STALE: {name} in robots/h1_2.py")
        print("[ok] frozen literals in robots/h1_2.py match the derivation")
        return

    print()
    print("# MuJoCo body order (DFS):")
    for i, n in enumerate(mj_bodies):
        print(f"#   {i:2d} {n}")
    print()
    print(emit(il_bodies, i2m_dof, m2i_dof, i2m_body, m2i_body))


if __name__ == "__main__":
    main()
