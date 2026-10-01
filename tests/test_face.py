"""Face matching against real photos. The portraits in tests/data are public
domain NASA photos from Wikimedia Commons (see tests/data/README.md)."""

import cv2
import numpy as np
import pytest

from cyberbullying.face import MATCH_THRESHOLD, NoFaceError, average, decode_image, similarity
from tests.conftest import image, jpeg, solid


def test_embeddings_are_unit_length(face_matcher):
    embedding = face_matcher.embed(image("sally_ride_1984.jpg"))
    assert embedding.shape == (128,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)


def test_two_photos_of_the_same_person_match(face_matcher):
    older = face_matcher.embed(image("sally_ride_1979.jpg"))
    newer = face_matcher.embed(image("sally_ride_1984.jpg"))
    assert similarity(older, newer) >= MATCH_THRESHOLD


def test_different_people_do_not_match(face_matcher):
    ride = face_matcher.embed(image("sally_ride_1984.jpg"))
    hadfield = face_matcher.embed(image("chris_hadfield_2011.jpg"))
    assert similarity(ride, hadfield) < MATCH_THRESHOLD


def test_a_mirrored_or_darker_photo_still_matches(face_matcher):
    photo = image("sally_ride_1984.jpg")
    enrolled = face_matcher.embed(photo)
    darker = cv2.convertScaleAbs(photo, alpha=0.7, beta=-10)
    assert similarity(enrolled, face_matcher.embed(cv2.flip(photo, 1))) >= MATCH_THRESHOLD
    assert similarity(enrolled, face_matcher.embed(darker)) >= MATCH_THRESHOLD


def test_large_images_are_handled(face_matcher):
    big = cv2.resize(image("chris_hadfield_2011.jpg"), None, fx=3, fy=3)
    small = face_matcher.embed(image("chris_hadfield_2011.jpg"))
    assert similarity(face_matcher.embed(big), small) >= MATCH_THRESHOLD


def test_no_face_raises(face_matcher):
    with pytest.raises(NoFaceError):
        face_matcher.embed(solid((128, 128, 128)))


def test_average_is_unit_length():
    a, b = np.array([1.0, 0.0], np.float32), np.array([0.0, 1.0], np.float32)
    mean = average([a, b])
    assert np.linalg.norm(mean) == pytest.approx(1.0)
    assert similarity(mean, a) == pytest.approx(similarity(mean, b))


def test_decode_image():
    assert decode_image(b"") is None
    assert decode_image(b"definitely not a picture") is None
    decoded = decode_image(jpeg(solid((0, 0, 255))))
    assert decoded.shape == (120, 160, 3)
