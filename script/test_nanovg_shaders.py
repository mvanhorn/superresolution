#!/usr/bin/env python3
"""Regression test for NanoVG RHI GLSL 4.30 compatibility.

Loads the production NanoVG vertex/fragment shaders, rejects any declared
GLSL version above 430, and compiles the on-disk sources with glslangValidator
for linkage plus the OpenGL and Vulkan 1.2 SPIR-V targets. Version directives
are never rewritten before compilation.

Supply a local validator with one of:
  python3 script/test_nanovg_shaders.py --validator /path/to/glslangValidator
  GLSLANG_VALIDATOR=/path/to/glslangValidator python3 script/test_nanovg_shaders.py
  python3 script/test_nanovg_shaders.py
    (uses glslangValidator or glslang from PATH)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
SHADER_DIR = REPO_ROOT / "common" / "src" / "main" / "resources" / "shader" / "nanovg"
VERT_SHADER = SHADER_DIR / "nanovg_rhi.vert.glsl"
FRAG_SHADER = SHADER_DIR / "nanovg_rhi.frag.glsl"
MAX_GLSL_VERSION = 430
SUBPROCESS_TIMEOUT_SEC = 30
SPIRV_MAGIC = 0x07230203
OP_NAME = 5
OP_DECORATE = 71
DECORATION_BINDING = 33
VERSION_RE = re.compile(r"^\s*#version\s+(\d+)", re.MULTILINE)

PREREQ_HINT = """glslangValidator is required but was not found.
Install it locally, then rerun with one of:
  python3 script/test_nanovg_shaders.py --validator /path/to/glslangValidator
  GLSLANG_VALIDATOR=/path/to/glslangValidator python3 script/test_nanovg_shaders.py
  python3 script/test_nanovg_shaders.py   # glslangValidator or glslang on PATH
A missing compiler is a prerequisite failure, not a passing result."""


class TestFailure(Exception):
    pass


class PrerequisiteFailure(Exception):
    pass


def declared_version(path: Path, source: str) -> int:
    match = VERSION_RE.search(source)
    if not match:
        raise TestFailure(f"{path} is missing a #version directive")
    return int(match.group(1))


def require_max_version(path: Path, source: str) -> int:
    version = declared_version(path, source)
    if version > MAX_GLSL_VERSION:
        raise TestFailure(
            f"{path} declares GLSL {version}, which exceeds the maximum of {MAX_GLSL_VERSION}"
        )
    return version


def usable_validator(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def resolve_validator(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if usable_validator(path):
            return path.resolve()
        raise PrerequisiteFailure(
            f"supplied validator is not an executable file: {path}\n\n{PREREQ_HINT}"
        )
    env_path = os.environ.get("GLSLANG_VALIDATOR")
    if env_path:
        path = Path(env_path).expanduser()
        if usable_validator(path):
            return path.resolve()
        raise PrerequisiteFailure(
            f"GLSLANG_VALIDATOR is not an executable file: {path}\n\n{PREREQ_HINT}"
        )
    for name in ("glslangValidator", "glslang"):
        found = shutil.which(name)
        if found:
            path = Path(found)
            if usable_validator(path):
                return path.resolve()
    raise PrerequisiteFailure(PREREQ_HINT)


def run_validator(validator: Path, args: list[str]) -> None:
    try:
        result = subprocess.run(
            [str(validator), *args],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TestFailure(
            f"glslangValidator timed out after {SUBPROCESS_TIMEOUT_SEC}s: {' '.join(args)}"
        ) from exc
    if result.returncode != 0:
        output = (result.stdout or "") + (result.stderr or "")
        raise TestFailure(
            f"glslangValidator failed ({result.returncode}) for {' '.join(args)}\n{output}"
        )


def spirv_bindings(path: Path) -> dict[str, int]:
    data = path.read_bytes()
    if len(data) < 20 or len(data) % 4 != 0:
        raise TestFailure(f"{path.name} is empty or not a SPIR-V binary")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if words[0] != SPIRV_MAGIC:
        raise TestFailure(f"{path.name} is missing the SPIR-V magic number")
    names: dict[int, str] = {}
    bindings: dict[int, int] = {}
    index = 5
    while index < len(words):
        count = words[index] >> 16
        opcode = words[index] & 0xFFFF
        if count == 0 or index + count > len(words):
            raise TestFailure(f"{path.name} has a truncated SPIR-V instruction")
        body = words[index + 1 : index + count]
        if opcode == OP_NAME and len(body) >= 2:
            raw = struct.pack(f"<{len(body) - 1}I", *body[1:])
            names[body[0]] = raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
        elif opcode == OP_DECORATE and len(body) >= 3 and body[1] == DECORATION_BINDING:
            bindings[body[0]] = body[2]
        index += count
    return {names.get(result_id, f"%{result_id}"): value for result_id, value in bindings.items()}


def require_bindings(label: str, path: Path, required: dict[str, int], extra_values: set[int]) -> None:
    if path.stat().st_size <= 0:
        raise TestFailure(f"{label} SPIR-V output is empty: {path}")
    bindings = spirv_bindings(path)
    values = set(bindings.values())
    missing = extra_values - values
    if missing:
        raise TestFailure(f"{label} SPIR-V {path.name} is missing bindings {sorted(missing)}: {bindings}")
    for name, expected in required.items():
        actual = bindings.get(name)
        if actual != expected:
            raise TestFailure(
                f"{label} SPIR-V {path.name} expected {name} at binding {expected}, got {actual} from {bindings}"
            )


def compile_stage(validator: Path, stage: str, source: Path, output: Path, target_args: list[str]) -> None:
    run_validator(
        validator,
        [*target_args, "-S", stage, "-o", str(output), str(source)],
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise TestFailure(f"SPIR-V output is missing or empty: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validator",
        help="Path to glslangValidator (overrides GLSLANG_VALIDATOR and PATH)",
    )
    args = parser.parse_args()

    for path in (VERT_SHADER, FRAG_SHADER):
        if not path.is_file():
            raise TestFailure(f"Missing NanoVG shader resource: {path}")

    vert_bytes = VERT_SHADER.read_bytes()
    frag_bytes = FRAG_SHADER.read_bytes()
    vert_source = vert_bytes.decode("utf-8")
    frag_source = frag_bytes.decode("utf-8")
    vert_version = require_max_version(VERT_SHADER, vert_source)
    frag_version = require_max_version(FRAG_SHADER, frag_source)
    validator = resolve_validator(args.validator)

    with tempfile.TemporaryDirectory(prefix="nanovg-shader-test-") as tmp:
        tmp_dir = Path(tmp)
        vert_copy = tmp_dir / "nanovg_rhi.vert"
        frag_copy = tmp_dir / "nanovg_rhi.frag"
        vert_copy.write_bytes(vert_bytes)
        frag_copy.write_bytes(frag_bytes)
        run_validator(validator, ["-l", str(vert_copy), str(frag_copy)])

        targets = {
            "opengl": ["--client", "opengl100", "--target-env", "opengl", "--target-env", "spirv1.4"],
            "vulkan1.2": ["--target-env", "vulkan1.2", "--target-env", "spirv1.4"],
        }
        for target_name, target_args in targets.items():
            vert_spv = tmp_dir / f"nanovg_rhi.vert.{target_name}.spv"
            frag_spv = tmp_dir / f"nanovg_rhi.frag.{target_name}.spv"
            compile_stage(validator, "vert", VERT_SHADER, vert_spv, target_args)
            compile_stage(validator, "frag", FRAG_SHADER, frag_spv, target_args)
            require_bindings(f"{target_name} vertex", vert_spv, {}, {0})
            require_bindings(
                f"{target_name} fragment",
                frag_spv,
                {"tex": 2, "fontTex": 3},
                {1, 2, 3},
            )

    print(
        f"NanoVG shaders OK (GLSL {vert_version}/{frag_version} <= {MAX_GLSL_VERSION}) "
        f"using {validator}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PrerequisiteFailure as exc:
        print(f"PREREQUISITE FAILURE: {exc}", file=sys.stderr)
        sys.exit(2)
    except TestFailure as exc:
        print(f"TEST FAILURE: {exc}", file=sys.stderr)
        sys.exit(1)
