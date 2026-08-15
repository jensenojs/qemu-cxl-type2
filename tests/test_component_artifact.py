#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import component_artifact as artifact


def file_entry(path: str, content: bytes, mode: int = 0o755) -> dict[str, object]:
    return {
        "path": path,
        "type": "file",
        "mode": f"{mode:04o}",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def manifest(files: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "component": "fixture",
        "source": {"repository": "https://example.invalid/fixture.git", "commit": "a" * 40},
        "build": {"profile_sha256": "b" * 64, "toolchain_image": "example@sha256:" + "c" * 64},
        "files": files,
        "dynamic_dependencies": [],
    }


def add_file(stream: tarfile.TarFile, name: str, content: bytes, mode: int = 0o755) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    info.mode = mode
    stream.addfile(info, io.BytesIO(content))


def add_symlink(stream: tarfile.TarFile, name: str, target: str) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.mode = 0o777
    info.linkname = target
    stream.addfile(info)


class ComponentArtifactTest(unittest.TestCase):
    def test_component_candidate_descriptor_has_no_schema_version(self) -> None:
        publish = (Path(__file__).resolve().parents[1] / "scripts/publish_component.sh").read_text()
        self.assertNotIn('"schema_version"', publish)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def tar(self, name: str, writer) -> Path:
        path = self.root / name
        with tarfile.open(path, "w") as stream:
            writer(stream)
        return path

    def test_valid_file_and_symlink_extract(self) -> None:
        content = b"server\n"
        files = [
            file_entry("bin/server", content),
            {"path": "guest/lib.so", "type": "symlink", "mode": "0777", "link_target": "lib.so.1"},
            file_entry("guest/lib.so.1", b"shim\n"),
        ]
        files.sort(key=lambda item: item["path"])
        archive = self.tar(
            "valid.tar",
            lambda stream: (
                add_file(stream, "bin/server", content),
                add_symlink(stream, "guest/lib.so", "lib.so.1"),
                add_file(stream, "guest/lib.so.1", b"shim\n"),
            ),
        )
        output = self.root / "output"
        artifact.verify_archive(archive, manifest(files), output)
        self.assertEqual((output / "bin/server").read_bytes(), content)
        self.assertEqual((output / "guest/lib.so").readlink().as_posix(), "lib.so.1")

    def test_qemu_profile_declares_minimal_runtime_files(self) -> None:
        profile = json.loads(
            (Path(__file__).resolve().parents[1] / "manifests/build-profile.json").read_text()
        )
        self.assertEqual(
            profile["outputs"],
            [
                {"path": "evidence/cuda-api/cxl_hetgpu.c", "type": "file", "mode": "0644"},
                {"path": "evidence/cuda-api/cxl_type2.c", "type": "file", "mode": "0644"},
                {
                    "path": "evidence/cuda-api/cxl_type2_gpu_cmd.h",
                    "type": "file",
                    "mode": "0644",
                },
                {"path": "bin/qemu-system-x86_64", "type": "file", "mode": "0755"},
                {"path": "lib/libaio.so.1t64", "type": "file", "mode": "0644"},
                {"path": "lib/libcapstone.so.4", "type": "file", "mode": "0644"},
                {"path": "libexec/qemu/libhotblocks.so", "type": "file", "mode": "0755"},
                {"path": "share/qemu/bios-256k.bin", "type": "file", "mode": "0644"},
                {"path": "share/qemu/kvmvapic.bin", "type": "file", "mode": "0644"},
                {"path": "share/qemu/linuxboot_dma.bin", "type": "file", "mode": "0644"},
            ],
        )

    def test_rejects_absolute_path(self) -> None:
        archive = self.tar("absolute.tar", lambda stream: add_file(stream, "/escape", b"x"))
        with self.assertRaises(ValueError):
            artifact.verify_archive(archive, manifest([file_entry("escape", b"x")]), None)

    def test_rejects_parent_path(self) -> None:
        archive = self.tar("parent.tar", lambda stream: add_file(stream, "../escape", b"x"))
        with self.assertRaises(ValueError):
            artifact.verify_archive(archive, manifest([file_entry("escape", b"x")]), None)

    def test_rejects_hardlink(self) -> None:
        def writer(stream: tarfile.TarFile) -> None:
            info = tarfile.TarInfo("hardlink")
            info.type = tarfile.LNKTYPE
            info.linkname = "target"
            stream.addfile(info)

        archive = self.tar("hardlink.tar", writer)
        with self.assertRaises(ValueError):
            artifact.verify_archive(archive, manifest([file_entry("hardlink", b"")]), None)

    def test_rejects_duplicate_path(self) -> None:
        def writer(stream: tarfile.TarFile) -> None:
            add_file(stream, "same", b"one")
            add_file(stream, "same", b"two")

        archive = self.tar("duplicate.tar", writer)
        with self.assertRaises(ValueError):
            artifact.verify_archive(archive, manifest([file_entry("same", b"one")]), None)

    def test_rejects_symlink_ancestor(self) -> None:
        def writer(stream: tarfile.TarFile) -> None:
            add_symlink(stream, "bin", "target")
            add_file(stream, "bin/server", b"x")

        archive = self.tar("ancestor.tar", writer)
        files = [
            {"path": "bin", "type": "symlink", "mode": "0777", "link_target": "target"},
            file_entry("bin/server", b"x"),
        ]
        with self.assertRaises(ValueError):
            artifact.verify_archive(archive, manifest(files), None)

    def test_rejects_tampered_content(self) -> None:
        archive = self.tar("tampered.tar", lambda stream: add_file(stream, "bin/server", b"wrong"))
        with self.assertRaises(ValueError):
            artifact.verify_archive(archive, manifest([file_entry("bin/server", b"right")]), None)

    def test_formal_backend_load_failure_stops_before_symbol_lookup(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "hw/cxl/cxl_hetgpu.c").read_text()
        load = source.index("g_cuda_lib_handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);")
        formal_failure = source.index("if (formal_case_strict && !g_cuda_lib_handle)", load)
        symbol_lookup = source.index("if (g_cuda_lib_handle)", formal_failure)
        rejected = source[formal_failure:symbol_lookup]

        self.assertIn("formal library load failed path=%s dlerror=%s", rejected)
        self.assertIn("return HETGPU_ERROR_NO_DEVICE;", rejected)


if __name__ == "__main__":
    unittest.main()
