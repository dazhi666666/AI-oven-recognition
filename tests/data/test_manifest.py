import hashlib
from io import BytesIO
from unittest.mock import patch

import pytest

from oven_stereo.data.manifest import acquire_assets, asset_path, read_manifest, verify_asset


def test_existing_data_verified_without_network(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"original")
    asset = {"path": "file", "size_bytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()}
    with patch("oven_stereo.data.manifest.urlopen", side_effect=AssertionError("不应联网")):
        acquire_assets({"assets": [asset]}, tmp_path)
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA256"):
        acquire_assets({"assets": [asset]}, tmp_path)
    assert path.read_bytes() == b"modified"


@pytest.mark.parametrize("relative", ["../outside", "C:/outside"])
def test_path_escape_rejected(tmp_path, relative):
    with pytest.raises(ValueError):
        asset_path(tmp_path, relative)


def test_missing_asset_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_asset(tmp_path / "missing", {"size_bytes": 0, "sha256": ""})


def test_wrong_manifest_version_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version": 99}')
    with pytest.raises(ValueError, match="schema_version"):
        read_manifest(path)


@pytest.mark.parametrize("valid_hash", [True, False])
@pytest.mark.parametrize("use_curl", [True, False])
def test_download_only_publishes_verified_file(tmp_path, valid_hash, use_curl):
    content = b"downloaded data"
    asset = {"path": "inputs/file", "url": "https://example.invalid/file", "size_bytes": len(content),
             "sha256": hashlib.sha256(content if valid_hash else b"wrong").hexdigest()}
    def fake_curl(args, *, stdout, check):
        assert args[0] == "curl.exe" and check
        stdout.write(content)

    with patch("oven_stereo.data.manifest.urlopen", return_value=BytesIO(content)), \
         patch("oven_stereo.data.manifest.shutil.which", return_value="curl.exe" if use_curl else None), \
         patch("oven_stereo.data.manifest.subprocess.run", side_effect=fake_curl):
        if valid_hash:
            acquire_assets({"assets": [asset]}, tmp_path)
            assert (tmp_path / "inputs/file").read_bytes() == content
        else:
            with pytest.raises(ValueError, match="SHA256"):
                acquire_assets({"assets": [asset]}, tmp_path)
            assert not (tmp_path / "inputs/file").exists()
    assert not (tmp_path / "inputs/file.part").exists()
