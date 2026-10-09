"""Documentation may evolve in Git while release archives retain strict integrity."""
from pathlib import Path
import shutil
import json
import hashlib
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class VerificationTests(unittest.TestCase):
    def test_documentation_edit_preserves_source_checks_but_not_archive_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / 'release'
            shutil.copytree(ROOT, checkout, ignore=shutil.ignore_patterns('.git', '__pycache__', '.venv*', 'outputs', 'figures', 'wandb'))
            # Establish fixture hashes independently of the repository's older release manifest.
            manifest_path = checkout / 'release_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['files'] = [entry for entry in manifest['files'] if (checkout / entry['path']).is_file()]
            for entry in manifest['files']:
                entry['sha256'] = hashlib.sha256((checkout / entry['path']).read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            def verify(*args):
                return subprocess.run([sys.executable, 'scripts/verify_package.py', *args], cwd=checkout, capture_output=True, text=True)
            self.assertEqual(verify().returncode, 0)
            with (checkout / 'README.md').open('a') as file:
                file.write('\nDocumentation edit.\n')
            result = verify()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Release file mismatch: README.md', result.stderr)
            self.assertEqual(verify('--source-only').returncode, 0)
            with (checkout / 'vendor/varc/src/ARC_ViT.py').open('a') as file:
                file.write('\n# Source changed.\n')
            result = verify('--source-only')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Hash mismatch:', result.stderr)


if __name__ == '__main__':
    unittest.main()
