import subprocess
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pull_component.sh"


class PullComponentCliTests(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["bash", str(SCRIPT), *args], text=True, capture_output=True)

    def test_runtime_is_required(self):
        result = self.run_script("--work-dir", "/unused", "/candidate", "/output")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--container-runtime is required", result.stderr)

    def test_work_dir_is_required(self):
        result = self.run_script("--container-runtime", "docker", "/candidate", "/output")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--work-dir is required", result.stderr)

    def test_build_and_pull_share_the_payload_verifier(self):
        root = SCRIPT.parents[1]
        build = (root / "scripts/build_component.sh").read_text()
        pull = SCRIPT.read_text()
        self.assertIn('scripts/verify_component_payload.sh" "$PAYLOAD"', build)
        self.assertIn("bash /verify_component_payload.sh /payload /work/evidence", pull)
        self.assertIn('print(c["toolchain_image"])', pull)


if __name__ == "__main__":
    unittest.main()
