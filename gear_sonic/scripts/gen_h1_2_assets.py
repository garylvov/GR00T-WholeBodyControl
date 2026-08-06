#!/usr/bin/env python3
# ADDED BY imprint (glvov) -- H1-2 embodiment for SONIC. Not upstream code.
"""Generate the SONIC H1-2 robot assets (MJCF + URDF) from our canonical model.

PROVENANCE / PROJECT LAW
------------------------
Our project law states that the USD factory
(``imprint/integrations/unitree_lab/h1_2_usd.py``, r13) is the single source of
truth for every H1-2 model, and that vendor URDF/MJCF are *not* authoritative
for mounts and frames.  The factory's own blessed no-fingers source is

    third_party/ProtoMotions/protomotions/data/assets/mjcf/h1_2_box_feet.xml

which the factory names ``DEFAULT_BOX_FEET_MJCF`` and describes as "the exact
PRIMITIVE-collision model the H1_2 tracker policy was trained on".  That file is
therefore the *input* to this generator, and everything SONIC loads is derived
from it -- no vendor URDF is used anywhere in this path.

WHAT THIS EMITS
---------------
1. ``data/assets/robot_description/mjcf/h1_2.xml``
   The canonical box-feet MJCF with the massless, geom-only ``head_aux`` frame
   removed.  ``head_aux`` is a zero-density 0.1 mm marker sphere welded to
   ``torso_link``; it carries no dynamics.  It is dropped because SONIC's
   ``IsaacLabMuJoCoConverter.convert()`` hard-assumes ``num_bodies == num_dof+1``
   when it auto-detects per-body tensors, and Isaac Lab's URDF importer merges
   fixed-joint links away anyway.  Its (constant) offset in the torso frame is
   re-exported as ``H1_2_HEAD_OFFSET_IN_TORSO`` in ``robots/h1_2.py`` so head
   tracking remains expressible without a body.

2. ``data/assets/robot_description/urdf/h1_2/h1_2.urdf``
   The same tree as a URDF: identical link names, identical joint names,
   identical parent/child structure, identical joint origins, axes and limits,
   identical inertials, and the identical PRIMITIVE collision set.  MuJoCo
   capsules are emitted as URDF cylinders because SONIC spawns with
   ``UrdfFileCfg(replace_cylinders_with_capsules=True)``, which turns them back
   into capsules inside Isaac.

   Visual geometry: by default the collision primitives are also emitted as
   visuals, so the articulation renders (as a capsule figure) with no mesh
   files shipped.  ``--with-meshes`` instead copies the 30 STLs referenced by
   the canonical MJCF into ``urdf/h1_2/meshes/`` and emits real mesh visuals.

The two files are verified to describe the same robot by
``gen_h1_2_order_tables.py`` and by ``tests/test_h1_2_embodiment.py``.

Usage
-----
    python gear_sonic/scripts/gen_h1_2_assets.py \
        --source /path/to/ProtoMotions/protomotions/data/assets/mjcf/h1_2_box_feet.xml
"""

from __future__ import annotations

import argparse
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT.parent
    / "ProtoMotions/protomotions/data/assets/mjcf/h1_2_box_feet.xml"
)
OUT_MJCF = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/h1_2.xml"
OUT_URDF_DIR = REPO_ROOT / "gear_sonic/data/assets/robot_description/urdf/h1_2"

# The one body we drop: a massless marker frame, see module docstring.
DROP_BODIES = ("head_aux",)


# --------------------------------------------------------------------------- #
# small math helpers (no numpy dependency so this runs in any interpreter)
# --------------------------------------------------------------------------- #
def quat_wxyz_to_rpy(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """MuJoCo (w, x, y, z) -> URDF fixed-axis roll/pitch/yaw (XYZ extrinsic)."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    w, x, y, z = w / n, x / n, y / n, z / n
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


def rpy_aligning_z_to(v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Fixed-axis rpy whose rotation maps +z onto the (non-zero) direction ``v``."""
    x, y, z = v
    n = math.sqrt(x * x + y * y + z * z)
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    x, y, z = x / n, y / n, z / n
    # Rz(yaw) @ Ry(pitch) applied to +z gives (sin p cos y, sin p sin y, cos p).
    pitch = math.acos(max(-1.0, min(1.0, z)))
    yaw = math.atan2(y, x)
    return (0.0, pitch, yaw)


def fmt(vals) -> str:
    return " ".join(f"{v:.10g}" for v in vals)


def floats(text: str | None, default):
    if text is None:
        return list(default)
    return [float(t) for t in text.split()]


# --------------------------------------------------------------------------- #
# MJCF parsing
# --------------------------------------------------------------------------- #
class MjBody:
    __slots__ = ("name", "pos", "quat", "parent", "children", "joint", "inertial", "geoms")

    def __init__(self, name):
        self.name = name
        self.pos = (0.0, 0.0, 0.0)
        self.quat = (1.0, 0.0, 0.0, 0.0)
        self.parent = None
        self.children = []
        self.joint = None  # dict or None (root free joint is dropped)
        self.inertial = None
        self.geoms = []


def parse_mjcf(path: Path) -> tuple[dict, MjBody, dict]:
    root = ET.parse(path).getroot()
    meshes = {
        m.get("name"): m.get("file")
        for m in root.findall("./asset/mesh")
    }
    bodies: dict[str, MjBody] = {}

    def walk(el, parent):
        for b in el.findall("body"):
            name = b.get("name")
            body = MjBody(name)
            body.pos = tuple(floats(b.get("pos"), (0, 0, 0)))
            body.quat = tuple(floats(b.get("quat"), (1, 0, 0, 0)))
            body.parent = parent
            bodies[name] = body
            if parent is not None:
                parent.children.append(body)
            for j in b.findall("joint"):
                if j.get("type") == "free":
                    continue
                lo, hi = floats(j.get("range"), (-3.14, 3.14))
                efflo, effhi = floats(j.get("actuatorfrcrange"), (-100.0, 100.0))
                body.joint = {
                    "name": j.get("name"),
                    "axis": tuple(floats(j.get("axis"), (0, 0, 1))),
                    "lower": lo,
                    "upper": hi,
                    "effort": max(abs(efflo), abs(effhi)),
                    "armature": float(j.get("armature", 0.0)),
                    "damping": float(j.get("damping", 0.0)),
                    "stiffness": float(j.get("stiffness", 0.0)),
                    "frictionloss": float(j.get("frictionloss", 0.0)),
                }
            it = b.find("inertial")
            if it is not None:
                body.inertial = {
                    "pos": tuple(floats(it.get("pos"), (0, 0, 0))),
                    "quat": tuple(floats(it.get("quat"), (1, 0, 0, 0))),
                    "mass": float(it.get("mass", 0.0)),
                    "diag": tuple(floats(it.get("diaginertia"), (0, 0, 0))),
                }
            for g in b.findall("geom"):
                body.geoms.append(dict(g.attrib))
            walk(b, body)

    wb = root.find("worldbody")
    walk(wb, None)
    root_body = next(b for b in bodies.values() if b.parent is None)
    return bodies, root_body, meshes


def prune(root_body: MjBody, drop: tuple[str, ...]) -> None:
    """Remove named bodies (and their subtrees) from the tree in place."""
    stack = [root_body]
    while stack:
        b = stack.pop()
        b.children = [c for c in b.children if c.name not in drop]
        stack.extend(b.children)


def dfs_order(root_body: MjBody):
    out = []

    def rec(b):
        out.append(b)
        for c in b.children:
            rec(c)

    rec(root_body)
    return out


# --------------------------------------------------------------------------- #
# emitters
# --------------------------------------------------------------------------- #
def write_mjcf(source: Path, out: Path, drop: tuple[str, ...], with_meshes: bool) -> None:
    """Copy the canonical MJCF, removing the dropped bodies, preserving order."""
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "h1_2")
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "body" and child.get("name") in drop:
                parent.remove(child)
    comp = root.find("compiler")
    if with_meshes:
        # meshdir is relative to the MJCF; point at the URDF's mesh dir.
        if comp is not None:
            comp.set("meshdir", "../urdf/h1_2/meshes/")
    else:
        # No STLs shipped -> drop the mesh assets and the (visual-only,
        # contype=0) mesh geoms so the file actually compiles in MuJoCo and
        # corresponds exactly, geom for geom, to the generated URDF.
        asset = root.find("asset")
        if asset is not None:
            root.remove(asset)
        for parent in root.iter():
            for child in list(parent):
                if child.tag == "geom" and child.get("type") == "mesh":
                    parent.remove(child)
        if comp is not None and "meshdir" in comp.attrib:
            del comp.attrib["meshdir"]
    ET.indent(tree, space="  ")
    header = (
        "<!-- GENERATED by gear_sonic/scripts/gen_h1_2_assets.py : do not edit by hand.\n"
        "     Source: ProtoMotions h1_2_box_feet.xml (the r13 USD factory's blessed\n"
        "     no-fingers, primitive-collision source, DEFAULT_BOX_FEET_MJCF).\n"
        "     Delta vs source: (a) the massless marker body 'head_aux' is removed so\n"
        "     that num_bodies == num_dof + 1, which SONIC's order converter requires;\n"
        "     (b) visual-only mesh geoms are dropped unless run with the meshes flag.\n"
        "     No kinematics, inertials, joint limits or collision geoms are altered. -->\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    body = ET.tostring(root, encoding="unicode")
    out.write_text(header + body + "\n")


def urdf_inertial(link: ET.Element, inertial) -> None:
    if inertial is None:
        # Isaac/URDF need a non-degenerate inertial on every link.
        el = ET.SubElement(link, "inertial")
        ET.SubElement(el, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(el, "mass", value="1e-6")
        ET.SubElement(
            el, "inertia", ixx="1e-9", ixy="0", ixz="0", iyy="1e-9", iyz="0", izz="1e-9"
        )
        return
    el = ET.SubElement(link, "inertial")
    ET.SubElement(
        el,
        "origin",
        xyz=fmt(inertial["pos"]),
        rpy=fmt(quat_wxyz_to_rpy(inertial["quat"])),
    )
    ET.SubElement(el, "mass", value=f"{inertial['mass']:.10g}")
    ixx, iyy, izz = inertial["diag"]
    ET.SubElement(
        el,
        "inertia",
        ixx=f"{ixx:.10g}",
        ixy="0",
        ixz="0",
        iyy=f"{iyy:.10g}",
        iyz="0",
        izz=f"{izz:.10g}",
    )


def geom_to_urdf(g: dict, meshes: dict, with_meshes: bool):
    """Return (origin_xyz, origin_rpy, geometry_element) or None if unsupported."""
    gtype = g.get("type", "sphere")
    if gtype == "mesh":
        if not with_meshes:
            return None
        f = meshes.get(g.get("mesh"))
        if f is None:
            return None
        geo = ET.Element("mesh", filename=f"meshes/{f}")
        return (floats(g.get("pos"), (0, 0, 0)), quat_wxyz_to_rpy(tuple(floats(g.get("quat"), (1, 0, 0, 0)))), geo)
    size = floats(g.get("size"), ())
    if gtype == "sphere":
        geo = ET.Element("sphere", radius=f"{size[0]:.10g}")
        return (floats(g.get("pos"), (0, 0, 0)), (0.0, 0.0, 0.0), geo)
    if gtype == "box":
        geo = ET.Element("box", size=fmt([2.0 * s for s in size[:3]]))
        return (
            floats(g.get("pos"), (0, 0, 0)),
            quat_wxyz_to_rpy(tuple(floats(g.get("quat"), (1, 0, 0, 0)))),
            geo,
        )
    if gtype in ("capsule", "cylinder"):
        ft = g.get("fromto")
        if ft is not None:
            a = floats(ft, ())[:3]
            b = floats(ft, ())[3:6]
            d = [b[i] - a[i] for i in range(3)]
            length = math.sqrt(sum(v * v for v in d))
            mid = [(a[i] + b[i]) / 2.0 for i in range(3)]
            rpy = rpy_aligning_z_to(tuple(d))
        else:
            length = 2.0 * size[1]
            mid = floats(g.get("pos"), (0, 0, 0))
            rpy = quat_wxyz_to_rpy(tuple(floats(g.get("quat"), (1, 0, 0, 0))))
        # Emitted as a cylinder on purpose: SONIC spawns with
        # UrdfFileCfg(replace_cylinders_with_capsules=True).
        geo = ET.Element("cylinder", radius=f"{size[0]:.10g}", length=f"{length:.10g}")
        return (mid, rpy, geo)
    return None


def write_urdf(root_body: MjBody, meshes: dict, out: Path, with_meshes: bool) -> None:
    robot = ET.Element("robot", name="h1_2")
    for b in dfs_order(root_body):
        link = ET.SubElement(robot, "link", name=b.name)
        urdf_inertial(link, b.inertial)
        for g in b.geoms:
            conv = geom_to_urdf(g, meshes, with_meshes)
            if conv is None:
                continue
            xyz, rpy, geo = conv
            is_collision = g.get("contype", "1") != "0"
            tags = ["collision"] if is_collision else ["visual"]
            if is_collision and not with_meshes:
                # No STLs shipped: render the collision primitives so the
                # articulation is still visible in the Isaac viewer.
                tags.append("visual")
            for tag in tags:
                el = ET.SubElement(link, tag)
                ET.SubElement(el, "origin", xyz=fmt(xyz), rpy=fmt(rpy))
                geom_el = ET.SubElement(el, "geometry")
                geom_el.append(ET.fromstring(ET.tostring(geo)))
        if b.parent is not None:
            j = b.joint
            joint = ET.SubElement(robot, "joint", name=j["name"], type="revolute")
            ET.SubElement(joint, "origin", xyz=fmt(b.pos), rpy=fmt(quat_wxyz_to_rpy(b.quat)))
            ET.SubElement(joint, "parent", link=b.parent.name)
            ET.SubElement(joint, "child", link=b.name)
            ET.SubElement(joint, "axis", xyz=fmt(j["axis"]))
            ET.SubElement(
                joint,
                "limit",
                lower=f"{j['lower']:.10g}",
                upper=f"{j['upper']:.10g}",
                effort=f"{j['effort']:.10g}",
                velocity="50",
            )
            ET.SubElement(
                joint, "dynamics", damping=f"{j['damping']:.10g}", friction=f"{j['frictionloss']:.10g}"
            )
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "<!-- GENERATED by gear_sonic/scripts/gen_h1_2_assets.py : do not edit by hand.\n"
        "     Source: ProtoMotions h1_2_box_feet.xml (the r13 USD factory's blessed\n"
        "     no-fingers, primitive-collision source, DEFAULT_BOX_FEET_MJCF).\n"
        "     Same link names / joint names / tree / origins / axes / limits / inertials\n"
        "     as ../../mjcf/h1_2.xml. Capsules become cylinders because SONIC\n"
        "     spawns with UrdfFileCfg(replace_cylinders_with_capsules=True). -->\n"
    )
    out.write_text(header + ET.tostring(robot, encoding="unicode") + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--mjcf-out", type=Path, default=OUT_MJCF)
    ap.add_argument("--urdf-dir", type=Path, default=OUT_URDF_DIR)
    ap.add_argument(
        "--with-meshes",
        action="store_true",
        help="copy the canonical STL visuals next to the URDF and reference them",
    )
    args = ap.parse_args()

    if not args.source.exists():
        raise SystemExit(f"canonical source MJCF not found: {args.source}")

    bodies, root_body, meshes = parse_mjcf(args.source)
    prune(root_body, DROP_BODIES)
    order = dfs_order(root_body)
    ndof = sum(1 for b in order if b.joint is not None)
    print(f"[gen] source={args.source}")
    print(f"[gen] bodies={len(order)} dof={ndof} (expect 28 / 27)")
    if len(order) != ndof + 1:
        raise SystemExit(f"invariant violated: num_bodies({len(order)}) != num_dof+1({ndof + 1})")

    write_mjcf(args.source, args.mjcf_out, DROP_BODIES, args.with_meshes)
    print(f"[gen] wrote {args.mjcf_out}")

    if args.with_meshes:
        mesh_src = args.source.parent.parent / "mesh/H1_2"
        mesh_dst = args.urdf_dir / "meshes"
        mesh_dst.mkdir(parents=True, exist_ok=True)
        for f in set(meshes.values()):
            src = mesh_src / f
            if src.exists():
                shutil.copy2(src, mesh_dst / f)
        print(f"[gen] copied {len(set(meshes.values()))} meshes -> {mesh_dst}")

    write_urdf(root_body, meshes, args.urdf_dir / "h1_2.urdf", args.with_meshes)
    print(f"[gen] wrote {args.urdf_dir / 'h1_2.urdf'}")


if __name__ == "__main__":
    main()
