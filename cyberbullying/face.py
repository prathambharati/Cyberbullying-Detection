"""Face matching with OpenCV's YuNet detector and SFace recognizer.

SFace turns a face into 128 numbers. Two photos of the same person end up
close together, so a login is a cosine similarity check against the numbers
saved when the user enrolled. No photos are stored, only those numbers.
"""

import threading

import cv2
import numpy as np

from . import downloads

# The cosine similarity cut-off the SFace authors recommend for "same person".
MATCH_THRESHOLD = 0.363

try:  # OpenCV's DNN module logs a harmless warning on every model load
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


class NoFaceError(ValueError):
    pass


class FaceMatcher:
    def __init__(self, detector_path, recognizer_path, min_confidence: float = 0.8):
        self._detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), min_confidence, 0.3, 5000)
        self._recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
        # One network, many web requests: OpenCV models aren't safe to run from two threads at once.
        self._lock = threading.Lock()

    @classmethod
    def from_downloads(cls) -> "FaceMatcher | None":
        """The matcher, or None when the models haven't been downloaded yet."""
        if not (downloads.is_available("face-detector") and downloads.is_available("face-recognizer")):
            return None
        return cls(downloads.ASSETS["face-detector"].path, downloads.ASSETS["face-recognizer"].path)

    def find_faces(self, image: np.ndarray) -> np.ndarray:
        """Each row is a box (x, y, w, h), five landmarks and a confidence."""
        height, width = image.shape[:2]
        with self._lock:
            self._detector.setInputSize((width, height))
            _, faces = self._detector.detect(image)
        return np.empty((0, 15), np.float32) if faces is None else faces

    def embed(self, image: np.ndarray) -> np.ndarray:
        image = _shrink(image)
        faces = self.find_faces(image)
        if len(faces) == 0:
            raise NoFaceError("No face in the image.")
        # With more than one person in view, use the biggest face, the one nearest the camera.
        face = max(faces, key=lambda row: row[2] * row[3])
        with self._lock:
            aligned = self._recognizer.alignCrop(image, face)
            embedding = self._recognizer.feature(aligned).flatten().astype(np.float32)
        return embedding / np.linalg.norm(embedding)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Both embeddings are already unit length."""
    return float(np.dot(a, b))


def average(embeddings: list[np.ndarray]) -> np.ndarray:
    mean = np.mean(embeddings, axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float32)


def decode_image(data: bytes) -> np.ndarray | None:
    """Uploaded bytes to an OpenCV image, or None if they aren't an image."""
    array = np.frombuffer(data, dtype=np.uint8)
    if array.size == 0:
        return None
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _shrink(image: np.ndarray, longest_side: int = 640) -> np.ndarray:
    height, width = image.shape[:2]
    scale = longest_side / max(height, width)
    if scale >= 1:
        return image
    size = (round(width * scale), round(height * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
