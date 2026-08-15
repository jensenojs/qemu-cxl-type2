#!/usr/bin/env python3
"""Create and verify the file identity of a CXL component artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tarfile
from typing import Any


MANIFEST_KEYS = {
    "component",
    "source",
    "build",
    "files",
    "dynamic_dependencies",
}
SOURCE_KEYS = {"repository", "commit"}
BUILD_KEYS = {"profile_sha256", "toolchain_image"}
FILE_KEYS = {"path", "type", "mode", "size", "sha256"}
SYMLINK_KEYS = {"path", "type", "mode", "link_target"}
DEPENDENCY_KEYS = {"path", "needed"}


def fail(message: str) -> None:
    raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        fail(f"{path}: expected a JSON object")
    return value


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_path(value: str) -> PurePosixPath:
    if not value or value.startswith("/"):
        fail(f"unsafe absolute or empty path: {value!r}")
    try:
        value.encode("utf-8", "strict")
    except UnicodeError as error:
        fail(f"path is not valid UTF-8: {value!r}: {error}")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        fail(f"unsafe path component: {value!r}")
    if path.as_posix() != value:
        fail(f"path is not normalized: {value!r}")
    return path


def resolved_symlink(path: PurePosixPath, target: str) -> PurePosixPath:
    if not target or target.startswith("/"):
        fail(f"unsafe symlink target for {path}: {target!r}")
    try:
        target.encode("utf-8", "strict")
    except UnicodeError as error:
        fail(f"symlink target is not valid UTF-8: {target!r}: {error}")
    parts = list(path.parent.parts)
    for part in PurePosixPath(target).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                fail(f"symlink escapes payload: {path} -> {target}")
            parts.pop()
        else:
            parts.append(part)
    if not parts:
        fail(f"symlink resolves to payload root: {path} -> {target}")
    return PurePosixPath(*parts)


def mode_string(mode: int) -> str:
    return f"{stat.S_IMODE(mode):04o}"


def needed_libraries(path: Path) -> list[str]:
    with path.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            return []
    result = subprocess.run(
        ["readelf", "-d", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    needed = []
    for line in result.stdout.splitlines():
        if "(NEEDED)" not in line or "[" not in line or "]" not in line:
            continue
        needed.append(line.rsplit("[", 1)[1].split("]", 1)[0])
    return sorted(set(needed))


def payload_facts(payload: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not payload.is_dir():
        fail(f"payload directory does not exist: {payload}")
    files: list[dict[str, Any]] = []
    dependencies: list[dict[str, Any]] = []
    for path in sorted(payload.rglob("*"), key=lambda item: item.relative_to(payload).as_posix()):
        relative = path.relative_to(payload).as_posix()
        normalized_path(relative)
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(path)
            resolved_symlink(PurePosixPath(relative), target)
            files.append(
                {
                    "path": relative,
                    "type": "symlink",
                    "mode": mode_string(metadata.st_mode),
                    "link_target": target,
                }
            )
            continue
        if not stat.S_ISREG(metadata.st_mode):
            fail(f"unsupported payload node: {relative}")
        files.append(
            {
                "path": relative,
                "type": "file",
                "mode": mode_string(metadata.st_mode),
                "size": metadata.st_size,
                "sha256": sha256_file(path),
            }
        )
        needed = needed_libraries(path)
        if needed:
            dependencies.append({"path": relative, "needed": needed})
    return files, dependencies


def expected_outputs(profile: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = profile.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        fail("build profile outputs must be a non-empty array")
    normalized: list[dict[str, Any]] = []
    for output in outputs:
        if not isinstance(output, dict):
            fail("build profile output must be an object")
        output_type = output.get("type")
        keys = {"path", "type", "mode"}
        if output_type == "symlink":
            keys.add("link_target")
        if set(output) != keys:
            fail(f"unexpected profile output fields: {output}")
        path = normalized_path(output["path"])
        if output_type not in {"file", "symlink"}:
            fail(f"unsupported profile output type: {output_type}")
        if not isinstance(output["mode"], str) or len(output["mode"]) != 4:
            fail(f"invalid output mode: {output['mode']!r}")
        if output_type == "symlink":
            resolved_symlink(path, output["link_target"])
        normalized.append(output)
    normalized.sort(key=lambda item: item["path"])
    if len({item["path"] for item in normalized}) != len(normalized):
        fail("duplicate build profile output path")
    return normalized


def verify_profile(payload: Path, profile: dict[str, Any]) -> None:
    actual_files, _ = payload_facts(payload)
    actual_shape = []
    for item in actual_files:
        shape = {key: item[key] for key in ("path", "type", "mode")}
        if item["type"] == "symlink":
            shape["link_target"] = item["link_target"]
        actual_shape.append(shape)
    if actual_shape != expected_outputs(profile):
        fail(f"payload shape does not match build profile: {actual_shape!r}")


def create_manifest(
    payload: Path,
    profile_path: Path,
    contract_path: Path,
    source_commit: str,
) -> dict[str, Any]:
    profile = load_json(profile_path)
    contract = load_json(contract_path)
    verify_profile(payload, profile)
    files, dependencies = payload_facts(payload)
    return {
        "component": contract["component"],
        "source": {
            "repository": contract["source_repository"],
            "commit": source_commit,
        },
        "build": {
            "profile_sha256": sha256_bytes(canonical_bytes(profile)),
            "toolchain_image": contract["toolchain_image"],
        },
        "files": files,
        "dynamic_dependencies": dependencies,
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    if set(manifest) != MANIFEST_KEYS:
        fail(f"unexpected manifest fields: {sorted(set(manifest) - MANIFEST_KEYS)}")
    if not isinstance(manifest["component"], str) or not manifest["component"]:
        fail("invalid component name")
    if not isinstance(manifest["source"], dict) or set(manifest["source"]) != SOURCE_KEYS:
        fail("invalid source identity")
    if not isinstance(manifest["build"], dict) or set(manifest["build"]) != BUILD_KEYS:
        fail("invalid build identity")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        fail("manifest files must be non-empty")
    paths = set()
    symlinks = set()
    for item in files:
        if not isinstance(item, dict):
            fail("manifest file entry must be an object")
        entry_type = item.get("type")
        expected_keys = FILE_KEYS if entry_type == "file" else SYMLINK_KEYS
        if set(item) != expected_keys:
            fail(f"invalid fields for {entry_type}: {item}")
        path = normalized_path(item["path"])
        if path.as_posix() in paths:
            fail(f"duplicate manifest path: {path}")
        paths.add(path.as_posix())
        if not isinstance(item["mode"], str) or len(item["mode"]) != 4:
            fail(f"invalid manifest mode: {item['mode']!r}")
        if entry_type == "file":
            if not isinstance(item["size"], int) or item["size"] < 0:
                fail(f"invalid file size: {item['size']!r}")
            if not isinstance(item["sha256"], str) or len(item["sha256"]) != 64:
                fail("invalid file sha256")
        elif entry_type == "symlink":
            resolved_symlink(path, item["link_target"])
            symlinks.add(path.as_posix())
        else:
            fail(f"unsupported manifest entry type: {entry_type}")
    for path in paths:
        parts = PurePosixPath(path).parts
        for index in range(1, len(parts)):
            if PurePosixPath(*parts[:index]).as_posix() in symlinks:
                fail(f"manifest path traverses symlink ancestor: {path}")
    dependencies = manifest["dynamic_dependencies"]
    if not isinstance(dependencies, list):
        fail("dynamic_dependencies must be an array")
    dependency_paths = set()
    for dependency in dependencies:
        if not isinstance(dependency, dict) or set(dependency) != DEPENDENCY_KEYS:
            fail(f"invalid dynamic dependency entry: {dependency}")
        if dependency["path"] not in paths or dependency["path"] in dependency_paths:
            fail(f"invalid dynamic dependency path: {dependency['path']}")
        if dependency["needed"] != sorted(set(dependency["needed"])):
            fail(f"dynamic dependency list is not canonical: {dependency}")
        dependency_paths.add(dependency["path"])


def archive_facts(archive: Path) -> tuple[list[dict[str, Any]], tarfile.TarFile]:
    stream = tarfile.open(archive, "r:")
    members = stream.getmembers()
    names = set()
    symlinks = set()
    facts: list[dict[str, Any]] = []
    for member in members:
        try:
            path = normalized_path(member.name)
        except Exception:
            stream.close()
            raise
        name = path.as_posix()
        if name in names:
            stream.close()
            fail(f"duplicate archive path: {name}")
        names.add(name)
        if member.mode & 0o7000:
            stream.close()
            fail(f"special permission bits are not allowed: {name}")
        if getattr(member, "sparse", None):
            stream.close()
            fail(f"sparse files are not allowed: {name}")
        if member.isfile():
            extracted = stream.extractfile(member)
            if extracted is None:
                stream.close()
                fail(f"cannot read archive member: {name}")
            content = extracted.read()
            facts.append(
                {
                    "path": name,
                    "type": "file",
                    "mode": f"{member.mode:04o}",
                    "size": member.size,
                    "sha256": sha256_bytes(content),
                }
            )
        elif member.issym():
            resolved_symlink(path, member.linkname)
            symlinks.add(name)
            facts.append(
                {
                    "path": name,
                    "type": "symlink",
                    "mode": f"{member.mode:04o}",
                    "link_target": member.linkname,
                }
            )
        else:
            stream.close()
            fail(f"unsupported archive entry type: {name}")
    for name in names:
        parts = PurePosixPath(name).parts
        for index in range(1, len(parts)):
            if PurePosixPath(*parts[:index]).as_posix() in symlinks:
                stream.close()
                fail(f"archive path traverses symlink ancestor: {name}")
    facts.sort(key=lambda item: item["path"])
    return facts, stream


def verify_archive(archive: Path, manifest: dict[str, Any], extract_dir: Path | None) -> None:
    validate_manifest(manifest)
    facts, stream = archive_facts(archive)
    if facts != manifest["files"]:
        stream.close()
        fail("archive file facts do not match manifest")
    if extract_dir is None:
        stream.close()
        return
    if extract_dir.exists():
        stream.close()
        fail(f"extract directory already exists: {extract_dir}")
    extract_dir.mkdir(parents=True)
    members = {member.name: member for member in stream.getmembers()}
    for item in facts:
        if item["type"] != "file":
            continue
        destination = extract_dir / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = stream.extractfile(members[item["path"]])
        if source is None:
            stream.close()
            fail(f"cannot extract file: {item['path']}")
        with destination.open("xb") as output:
            shutil.copyfileobj(source, output)
        destination.chmod(int(item["mode"], 8))
    for item in facts:
        if item["type"] != "symlink":
            continue
        destination = extract_dir / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(item["link_target"])
    stream.close()
    actual_files, actual_dependencies = payload_facts(extract_dir)
    if actual_files != manifest["files"]:
        fail("restored payload facts do not match manifest")
    if actual_dependencies != manifest["dynamic_dependencies"]:
        fail("restored ELF dependencies do not match manifest")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser("verify-profile")
    profile_parser.add_argument("--payload", type=Path, required=True)
    profile_parser.add_argument("--profile", type=Path, required=True)

    create_parser = subparsers.add_parser("create-manifest")
    create_parser.add_argument("--payload", type=Path, required=True)
    create_parser.add_argument("--profile", type=Path, required=True)
    create_parser.add_argument("--contract", type=Path, required=True)
    create_parser.add_argument("--source-commit", required=True)
    create_parser.add_argument("--output", type=Path, required=True)

    archive_parser = subparsers.add_parser("verify-archive")
    archive_parser.add_argument("--archive", type=Path, required=True)
    archive_parser.add_argument("--manifest", type=Path, required=True)
    archive_parser.add_argument("--extract-dir", type=Path)

    payload_parser = subparsers.add_parser("verify-payload")
    payload_parser.add_argument("--payload", type=Path, required=True)
    payload_parser.add_argument("--manifest", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "verify-profile":
        verify_profile(args.payload, load_json(args.profile))
    elif args.command == "create-manifest":
        manifest = create_manifest(
            args.payload,
            args.profile,
            args.contract,
            args.source_commit,
        )
        validate_manifest(manifest)
        write_json(args.output, manifest)
    elif args.command == "verify-archive":
        verify_archive(args.archive, load_json(args.manifest), args.extract_dir)
    elif args.command == "verify-payload":
        manifest = load_json(args.manifest)
        validate_manifest(manifest)
        files, dependencies = payload_facts(args.payload)
        if files != manifest["files"] or dependencies != manifest["dynamic_dependencies"]:
            fail("payload facts do not match manifest")


if __name__ == "__main__":
    main()
