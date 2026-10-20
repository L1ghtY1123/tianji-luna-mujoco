"""Bounded, auditable URDF axis adaptation for the pinned cuRobo parser.

cuRobo 8e734f3 selects JointType only for exact signed coordinate axes. CAD
export roundoff is approximated here, in project-owned derived assets, rather
than changing the SDK or the authoritative URDF. The bound is Euclidean on
original components, before any normalization; arbitrary tilted axes fail.
"""

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from xml.parsers import expat

AXIS_TOLERANCE_L2 = 1e-6
SDK_COMMIT = "8e734f3ced1df898990bcd92de40abce475907db"
ADAPTER_VERSION = "signed-coordinate-axis-v1"
_ATTRIBUTE = re.compile(rb"\s+([^\s=<>]+)\s*=\s*([\"'])(.*?)\2", re.DOTALL)


def _tag_end(data, start):
    quote = None
    for position in range(start, len(data)):
        byte = data[position]
        if quote is not None:
            if byte == quote:
                quote = None
        elif byte in (34, 39):
            quote = byte
        elif byte == 62:
            return position + 1
    raise ValueError("unterminated URDF axis element")


def _coordinate_axis(text, joint_name):
    try:
        values = [float(item) for item in text.split()]
    except ValueError as error:
        raise ValueError(f"invalid URDF axis for {joint_name}: {text}") from error
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid URDF axis for {joint_name}: {text}")
    dominant = max(range(3), key=lambda index: abs(values[index]))
    target = [0, 0, 0]
    target[dominant] = 1 if values[dominant] >= 0 else -1
    difference = math.dist(values, target)
    if difference > AXIS_TOLERANCE_L2:
        raise ValueError(
            f"unsupported URDF axis for {joint_name}: {text}; "
            f"distance {difference:.12g} exceeds {AXIS_TOLERANCE_L2} from a signed coordinate axis"
        )
    return values, target, difference


def _regularized_bytes(source):
    # Expat provides byte positions, so comments, meshes, origins, limits, mimic
    # chains and formatting are retained exactly outside the changed xyz values.
    source.decode("utf-8-sig")
    parser = expat.ParserCreate()
    stack = []
    joint = None
    axis_count = 0
    edits, changes = [], []

    def start(name, attributes):
        nonlocal joint, axis_count
        stack.append(name)
        if len(stack) == 1 and name != "robot":
            raise ValueError("URDF root must be robot")
        if stack == ["robot", "joint"]:
            joint = attributes
            axis_count = 0
        if stack != ["robot", "joint", "axis"] or joint.get("type") == "fixed":
            return
        axis_count += 1
        if axis_count != 1 or joint.get("type") not in ("revolute", "continuous", "prismatic"):
            raise ValueError("unsupported or duplicate URDF joint axis")
        text = attributes.get("xyz")
        if text is None:
            raise ValueError("URDF axis must contain xyz")
        values, target, difference = _coordinate_axis(text, joint.get("name", "<unnamed>"))
        if difference == 0:
            return
        position = parser.CurrentByteIndex
        tag = source[position : _tag_end(source, position)]
        match = None
        for attribute in _ATTRIBUTE.finditer(tag):
            if attribute.group(1) == b"xyz":
                match = attribute
                break
        if match is None:
            raise ValueError("URDF axis xyz byte span is unavailable")
        replacement = " ".join(str(value) for value in target).encode("ascii")
        edits.append((position + match.start(3), position + match.end(3), replacement))
        changes.append(
            {
                "joint_name": joint.get("name"),
                "joint_type": joint["type"],
                "source_xyz_text": text,
                "source_xyz": values,
                "derived_xyz": target,
                "difference_l2": difference,
            }
        )

    def end(name):
        stack.pop()

    def reject_doctype(*args):
        raise ValueError("URDF document type declarations are not supported")

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.StartDoctypeDeclHandler = reject_doctype
    try:
        parser.Parse(source, True)
    except expat.ExpatError as error:
        raise ValueError("invalid URDF XML") from error
    derived = source
    for begin, end, replacement in reversed(edits):
        derived = derived[:begin] + replacement + derived[end:]
    return derived, changes


def _check_cache_path(path, root, source_directory):
    # Check every existing component before following it for cache I/O. Checking
    # only the root permits a version/hash junction to lead back into the model.
    if not path.is_relative_to(root):
        raise ValueError("URDF cache path redirects outside its root")
    for component in (path, *path.parents):
        if component.is_symlink() or (os.name == "nt" and component.is_junction()):
            raise ValueError(f"URDF cache path redirects through a link: {component}")
        if component == root:
            break
    resolved = path.resolve()
    if (
        resolved != path
        or not resolved.is_relative_to(root)
        or resolved.is_relative_to(source_directory)
    ):
        raise ValueError(f"URDF cache path redirects outside its permitted directory: {path}")
    if path.is_file() and path.stat().st_nlink > 1:
        raise ValueError(f"URDF cache file redirects through a hard link: {path}")


def _write_atomic(path, data, root, source_directory):
    _check_cache_path(path, root, source_directory)
    if path.is_file() and path.read_bytes() == data:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        _check_cache_path(path, root, source_directory)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def regularize_joint_axes(urdf_path, *, cache_root=None):
    """Return a source path or a content-verified derived file in a writable cache.

    Missing axis elements retain the URDF default +X. Fixed-joint axes are
    ignored. No rewrite/cache is needed for files whose moving axes are already
    exact. A changed file has an adjacent .axis-normalization.json audit record.
    """
    path = Path(urdf_path).expanduser().resolve()
    source = path.read_bytes()
    derived, changes = _regularized_bytes(source)
    if not changes:
        return path
    configured = cache_root or os.environ.get("GAP_CUROBO_URDF_CACHE")
    root = (
        Path(configured) if configured else Path(tempfile.gettempdir()) / "gap-tianji-curobo-urdf"
    )
    root = root.expanduser().resolve()
    if root == path.parent or root.is_relative_to(path.parent):
        raise ValueError("URDF cache must be outside the source directory")
    source_hash = hashlib.sha256(source).hexdigest()
    location_hash = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    directory = root / ADAPTER_VERSION / (source_hash + "-" + location_hash)
    _check_cache_path(directory, root, path.parent)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "robot.urdf"
    audit = destination.with_suffix(".axis-normalization.json")
    # Validate both outputs before reading or creating either artifact.
    _check_cache_path(destination, root, path.parent)
    _check_cache_path(audit, root, path.parent)
    manifest = {
        "schema_version": 1,
        "adapter": ADAPTER_VERSION,
        "sdk_commit": SDK_COMMIT,
        "source_path": str(path),
        "derived_path": str(destination),
        "source_sha256": source_hash,
        "derived_sha256": hashlib.sha256(derived).hexdigest(),
        "axis_tolerance_l2": AXIS_TOLERANCE_L2,
        "metric": "Euclidean distance of original xyz to signed coordinate axis; no prior rescaling",
        "changes": changes,
    }
    _write_atomic(destination, derived, root, path.parent)
    _write_atomic(audit, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"), root, path.parent)
    return destination
