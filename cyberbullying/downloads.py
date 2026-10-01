"""Fetches the files that are too big for git or belong to someone else.

    python -m cyberbullying.downloads             # face and eye models
    python -m cyberbullying.downloads training    # datasets and GloVe, for retraining

Every file is checked against a SHA-256 hash before it's kept, so a changed
or broken download fails loudly instead of quietly giving different results.
"""

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

from .paths import DATA_DIR, VISION_DIR

JIGSAW = "https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge/resolve/main"
OLID = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/offensive"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    sha256: str
    path: Path


def _asset(name, url, sha256, folder, filename=None):
    return Asset(name, url, sha256, folder / (filename or url.rsplit("/", 1)[-1]))


ASSETS = {
    asset.name: asset
    for asset in [
        # Training data
        _asset("tweets", "https://huggingface.co/datasets/poorvanshi04/cyberbullying_tweets.csv/resolve/main/cyberbullying_tweets.csv",
               "0310ff5139cc3d7055c0a2b10f7e9253dfabc0bcc83e85dad9b2e074b29f8c90", DATA_DIR),
        _asset("jigsaw-train", f"{JIGSAW}/train.csv",
               "bd4084611bd27c939ba98e5e63bc3e5a2c1a4e99477dcba46c829e4c986c429d", DATA_DIR, "jigsaw_train.csv"),
        _asset("jigsaw-test", f"{JIGSAW}/test.csv",
               "c2513ce4abb98c4d1d216e3ca0d4377d57589a0989aa8c06a840509a16c786e8", DATA_DIR, "jigsaw_test.csv"),
        _asset("jigsaw-test-labels", f"{JIGSAW}/test_labels.csv",
               "2a56dcbeba5c05f965a636f56cb5ae972bad60c3b952c239b49be18d7ab70f49", DATA_DIR, "jigsaw_test_labels.csv"),
        _asset("olid-train-text", f"{OLID}/train_text.txt",
               "78a7a32e38b10af7d8970b008bf17f661c8d0a90dad145fa0fa6a944669650db", DATA_DIR, "olid_train_text.txt"),
        _asset("olid-train-labels", f"{OLID}/train_labels.txt",
               "c0b7d6ebdaa4ebcf6fc557ef1e775d92eda160218a0e3b1dd48eb8234dc892a6", DATA_DIR, "olid_train_labels.txt"),
        _asset("olid-val-text", f"{OLID}/val_text.txt",
               "816f36d180c35f15a5104838cb73856a0bef42043482fe738f3481b06242a55c", DATA_DIR, "olid_val_text.txt"),
        _asset("olid-val-labels", f"{OLID}/val_labels.txt",
               "ed2deb776bd1c52fb8221fadd3360e32d9dfe46842d78053528126e46363a258", DATA_DIR, "olid_val_labels.txt"),
        _asset("olid-test-text", f"{OLID}/test_text.txt",
               "25b08c3333c26190f1023961c4508ec9aab24d4722b1a3ea7a6040724c120547", DATA_DIR, "olid_test_text.txt"),
        _asset("olid-test-labels", f"{OLID}/test_labels.txt",
               "41d05a7aa0b01f5dafab21b95adb4f979cb4226c046ff315702774d10dac1605", DATA_DIR, "olid_test_labels.txt"),
        _asset("hate-speech", "https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/resolve/main/measuring-hate-speech.parquet",
               "6819525ce61bc24344df9fc3f7bf48270b31038273cc27c67fc225b51433b0e1", DATA_DIR),
        _asset("glove", "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz",
               "4934d4708bf3d65f28fc984dbbfa18af9cbd2505d166a7258a1a957808559b43", DATA_DIR),
        # Webcam features
        _asset("face-detector", f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
               "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", VISION_DIR),
        _asset("face-recognizer", f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
               "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79", VISION_DIR),
        _asset("eye-tracker", "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
               "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff", VISION_DIR),
    ]
}

GROUPS = {
    "vision": ["face-detector", "face-recognizer", "eye-tracker"],
    "training": [name for name, asset in ASSETS.items() if asset.path.parent == DATA_DIR],
}


class DownloadError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_available(name: str) -> bool:
    return ASSETS[name].path.exists()


def fetch(name: str, quiet: bool = False) -> Path:
    """Return the local path of an asset, downloading it the first time."""
    asset = ASSETS[name]
    if asset.path.exists():
        return asset.path

    asset.path.parent.mkdir(parents=True, exist_ok=True)
    partial = asset.path.with_name(asset.path.name + ".part")
    if not quiet:
        print(f"Downloading {asset.path.name}...", flush=True)
    try:
        with requests.get(asset.url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(partial, "wb") as f:
                for block in response.iter_content(chunk_size=1 << 20):
                    f.write(block)
    except requests.RequestException as exc:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"Couldn't download {name} from {asset.url}: {exc}") from exc

    actual = sha256_of(partial)
    if actual != asset.sha256:
        partial.unlink(missing_ok=True)
        raise DownloadError(
            f"The {name} download doesn't match its checksum (got {actual}, "
            f"expected {asset.sha256}). The file may have changed upstream."
        )
    partial.replace(asset.path)
    return asset.path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download model and data files.")
    parser.add_argument(
        "names",
        nargs="*",
        metavar="NAME",
        help=f"'vision' (the default), 'training', or single files: {', '.join(ASSETS)}",
    )
    requested = parser.parse_args(argv).names or ["vision"]
    names = []
    for name in requested:
        if name in GROUPS:
            names.extend(GROUPS[name])
        elif name in ASSETS:
            names.append(name)
        else:
            parser.error(f"unknown name: {name}")

    for name in names:
        try:
            path = fetch(name)
        except DownloadError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"{name:20} ready at {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
