import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


class ComponentScriptTests(unittest.TestCase):
    def run_script(self, name: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPTS / name), *args],
            text=True,
            capture_output=True,
        )

    def test_development_contract_selects_persistent_managed_entrypoint(self) -> None:
        contract = json.loads((ROOT / "manifests/artifact-contract.json").read_text())
        self.assertEqual(
            contract["development_execution"],
            {
                "entrypoint": "scripts/run_local_component.sh",
                "profile": "manifests/build-profile.json",
                "inputs": [],
                "output_manifest": "component/outputs/manifest.json",
                "build_root": "persistent",
            },
        )

    def test_component_entrypoints_require_explicit_arguments(self) -> None:
        for script in (
            "build_component.sh",
            "package_component.sh",
            "publish_component.sh",
            "run_local_component.sh",
        ):
            with self.subTest(script=script):
                result = self.run_script(script)
                self.assertEqual(result.returncode, 2)
                self.assertIn("usage:", result.stderr)

    def test_build_reads_prepared_gitlink_without_mutating_source(self) -> None:
        build = (SCRIPTS / "build_component.sh").read_text()
        pipeline = (ROOT / ".cnb.yml").read_text()
        self.assertNotIn("submodule update", build)
        self.assertIn("git submodule update --init --depth 1 subprojects/hetGPU", pipeline)

    def test_existing_execution_root_is_rejected_without_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execution = root / "execution"
            execution.mkdir()
            sentinel = execution / "sentinel"
            sentinel.write_text("preserve\n")
            cache = root / "cache"
            cache.mkdir()
            build = root / "build"
            build.mkdir()

            before = sorted(path.name for path in execution.iterdir())
            result = self.run_script(
                "run_local_component.sh",
                "--profile",
                str(ROOT / "manifests/build-profile.json"),
                "--execution-root",
                str(execution),
                "--cache-root",
                str(cache),
                "--build-root",
                str(build),
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("execution root must not exist", result.stderr)
            self.assertEqual(sentinel.read_text(), "preserve\n")
            self.assertEqual(sorted(path.name for path in execution.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
