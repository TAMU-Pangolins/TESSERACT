import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from nucres.data import (
    DATASET_SENTINEL,
    ensure_hfb_dataset,
    expected_hfb_filenames,
)
from nucres.hfb_adapter import resolve_density_paths


def build_test_archive(root: Path) -> tuple[Path, str]:
    source = root / "source"
    source.mkdir()
    for name in expected_hfb_filenames():
        (source / name).write_bytes(name.encode("ascii"))

    archive = root / "hfb-test.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(source.iterdir()):
            bundle.write(path, arcname=path.name)
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


class HostedDataTest(unittest.TestCase):
    def test_ensure_hfb_dataset_downloads_complete_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            archive, digest = build_test_archive(temp)
            cache = temp / "cache"

            installed = ensure_hfb_dataset(
                cache,
                archive_url=archive.as_uri(),
                archive_sha256=digest,
                show_progress=False,
            )

            self.assertEqual(installed, cache)
            self.assertTrue((cache / "z008.tab").is_file())
            self.assertTrue((cache / "z100.cor").is_file())
            self.assertTrue((cache / "z109.tab").is_file())
            self.assertTrue((cache / DATASET_SENTINEL).is_file())

    def test_complete_dataset_skips_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            for name in expected_hfb_filenames():
                (cache / name).write_bytes(b"present")

            with patch("nucres.data.download_file") as download:
                installed = ensure_hfb_dataset(cache)

            self.assertEqual(installed, cache)
            download.assert_not_called()

    def test_resolver_installs_dataset_before_lookup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            tab = cache / "z014.tab"
            cor = cache / "z014.cor"

            def fake_ensure(root):
                self.assertEqual(root, cache)
                tab.write_bytes(b"tab")
                cor.write_bytes(b"cor")
                return cache

            with patch("nucres.hfb_adapter.ensure_hfb_dataset", side_effect=fake_ensure):
                resolved_tab, resolved_cor = resolve_density_paths(14, data_root=cache)

            self.assertEqual(resolved_tab, tab)
            self.assertEqual(resolved_cor, cor)


if __name__ == "__main__":
    unittest.main()
