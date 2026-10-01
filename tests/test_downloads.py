import hashlib

import pytest
import requests

from cyberbullying import downloads

CONTENT = b"pretend this is a model file"


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for start in range(0, len(self.content), 7):
            yield self.content[start : start + 7]


@pytest.fixture
def asset(tmp_path, monkeypatch):
    item = downloads.Asset("thing", "https://example.com/thing.bin", hashlib.sha256(CONTENT).hexdigest(),
                           tmp_path / "models" / "thing.bin")
    monkeypatch.setitem(downloads.ASSETS, "thing", item)
    return item


def test_download_is_checked_and_saved(asset, monkeypatch):
    monkeypatch.setattr(downloads.requests, "get", lambda url, stream, timeout: FakeResponse(CONTENT))
    assert downloads.fetch("thing", quiet=True) == asset.path
    assert asset.path.read_bytes() == CONTENT
    assert downloads.is_available("thing")


def test_a_file_that_is_already_there_is_not_downloaded_again(asset, monkeypatch):
    asset.path.parent.mkdir(parents=True)
    asset.path.write_bytes(CONTENT)

    def fail(*args, **kwargs):
        raise AssertionError("should not download")

    monkeypatch.setattr(downloads.requests, "get", fail)
    assert downloads.fetch("thing", quiet=True) == asset.path


def test_a_bad_checksum_leaves_nothing_behind(asset, monkeypatch):
    monkeypatch.setattr(downloads.requests, "get", lambda url, stream, timeout: FakeResponse(b"tampered"))
    with pytest.raises(downloads.DownloadError, match="checksum"):
        downloads.fetch("thing", quiet=True)
    assert list(asset.path.parent.iterdir()) == []


def test_network_errors_become_download_errors(asset, monkeypatch):
    def offline(*args, **kwargs):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(downloads.requests, "get", offline)
    with pytest.raises(downloads.DownloadError, match="Couldn't download"):
        downloads.fetch("thing", quiet=True)
    assert not asset.path.exists()


def test_every_asset_has_a_real_looking_checksum():
    for asset in downloads.ASSETS.values():
        assert len(asset.sha256) == 64 and int(asset.sha256, 16) >= 0
        assert asset.url.startswith("https://")


def test_groups_only_name_known_assets():
    for names in downloads.GROUPS.values():
        assert names and all(name in downloads.ASSETS for name in names)
    assert "glove" in downloads.GROUPS["training"]
    assert "face-recognizer" in downloads.GROUPS["vision"]


def test_cli_rejects_unknown_names(capsys):
    with pytest.raises(SystemExit):
        downloads.main(["not-a-thing"])
    assert "unknown name" in capsys.readouterr().err
