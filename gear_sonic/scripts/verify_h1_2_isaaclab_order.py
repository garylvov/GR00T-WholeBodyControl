#!/usr/bin/env python3
# ADDED BY imprint (glvov) -- H1-2 embodiment for SONIC. Not upstream code.
"""Runtime confirmation of the H1-2 Isaac Lab joint/body order.

Upstream is emphatic (``docs/source/references/conventions.md``): never trust a
hand-derived Isaac Lab order, print ``robot.joint_names`` at runtime.  Our
tables are derived by rule (BFS + alphabetical siblings) and already agree with
the *independently measured* order frozen in our r13 USD factory, but this
script is the last word: it spawns one env, loads ``H1_2_CFG`` through the same
``UrdfFileCfg`` the trainer uses, and asserts the runtime order equals the
frozen literals.

This is upstream gotcha #5 ("test with num_envs=1 first") in script form: a
wrong body name fails here with Isaac Lab naming the offending body, before any
training run is launched.

Run inside the groot-sonic-gpu env, from the repo root:

    python gear_sonic/scripts/verify_h1_2_isaaclab_order.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    from isaacsim import SimulationApp  # noqa: PLC0415

    app = SimulationApp({"headless": True})
    try:
        import isaaclab.sim as sim_utils  # noqa: PLC0415
        from isaaclab.assets import Articulation  # noqa: PLC0415

        sys.path.insert(0, str(REPO_ROOT))
        from gear_sonic.envs.manager_env.robots.h1_2 import (  # noqa: PLC0415
            H1_2_ACTION_SCALE,
            H1_2_CFG,
            H1_2_ISAACLAB_JOINTS,
            H1_2_ISAACLAB_TO_MUJOCO_BODY,
            H1_2_ISAACLAB_TO_MUJOCO_DOF,
            H1_2_MUJOCO_TO_ISAACLAB_BODY,
            H1_2_MUJOCO_TO_ISAACLAB_DOF,
        )
        from gear_sonic.trl.utils.order_converter import H1_2Converter  # noqa: PLC0415

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cpu"))
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        robot = Articulation(H1_2_CFG.replace(prim_path="/World/Robot"))
        sim.reset()

        rt_joints = list(robot.joint_names)
        rt_bodies = list(robot.body_names)
        expected_joints = H1_2_ISAACLAB_JOINTS[1:]  # body i+1 is driven by joint i

        print("\nRUNTIME robot.joint_names:")
        for i, n in enumerate(rt_joints):
            print(f"  {i:2d} {n}")
        print("\nRUNTIME robot.body_names:")
        for i, n in enumerate(rt_bodies):
            print(f"  {i:2d} {n}")

        failures = []
        if rt_bodies != H1_2_ISAACLAB_JOINTS:
            failures.append(
                "BODY ORDER MISMATCH\n"
                f"  runtime : {rt_bodies}\n  frozen  : {H1_2_ISAACLAB_JOINTS}"
            )
        # The frozen table is a body list; the joint list must be the same
        # sequence with '_link' -> '_joint'. Compare by position, not by name
        # rewriting, so an unexpected naming scheme is reported rather than hidden.
        if len(rt_joints) != len(expected_joints):
            failures.append(f"DOF COUNT: runtime {len(rt_joints)} vs frozen {len(expected_joints)}")
        else:
            for i, (j, b) in enumerate(zip(rt_joints, expected_joints)):
                if j.replace("_joint", "") != b.replace("_link", ""):
                    failures.append(f"JOINT/BODY ORDER DIVERGES at index {i}: {j!r} vs body {b!r}")

        # Actuator coverage: every DOF must be claimed by exactly one group.
        missing = [n for n in rt_joints if n not in H1_2_ACTION_SCALE]
        if missing:
            failures.append(f"ACTION SCALE missing entries for: {missing}")

        conv = H1_2Converter()
        for label, arr, want in [
            ("ISAACLAB_TO_MUJOCO_DOF", H1_2_ISAACLAB_TO_MUJOCO_DOF, len(rt_joints)),
            ("MUJOCO_TO_ISAACLAB_DOF", H1_2_MUJOCO_TO_ISAACLAB_DOF, len(rt_joints)),
            ("ISAACLAB_TO_MUJOCO_BODY", H1_2_ISAACLAB_TO_MUJOCO_BODY, len(rt_bodies)),
            ("MUJOCO_TO_ISAACLAB_BODY", H1_2_MUJOCO_TO_ISAACLAB_BODY, len(rt_bodies)),
        ]:
            if len(arr) != want:
                failures.append(f"{label}: length {len(arr)} != runtime count {want}")
        print(f"\nconverter.num_dof              = {conv.num_dof}")
        print(f"converter.vr_3points_mujoco_indices = {conv.vr_3points_mujoco_indices}")
        print(f"converter.foot_mujoco_indices       = {conv.foot_mujoco_indices}")

        print()
        if failures:
            for f in failures:
                print("[FAIL] " + f)
            return 1
        print("[ok] runtime Isaac Lab order matches the frozen H1-2 tables")
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
