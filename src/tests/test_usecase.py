import numpy as np
import pytest

from dataclass import Keypoint
from usecase import (
    Midpoint,
    get_body_orientation,
    get_moved_degree,
    get_screen_orientation,
    normalize_degree,
    polar_to_xy,
)


def create_midpoint(cood: np.ndarray) -> Midpoint:
    return Midpoint(
        Keypoint(xy=cood, confidence=np.array(1)),
        Keypoint(xy=cood, confidence=np.array(1)),
    )


def test_normalize_degree():
    assert np.allclose(normalize_degree(-180.0), 180)
    assert np.allclose(normalize_degree(-90.0), 270)
    assert np.allclose(normalize_degree(0.0), 0)


class TestGetScreenOrientation:
    @pytest.mark.parametrize(
        "current_middle_hip_point, expected",
        [
            ([0, -1], 0.0),
            ([-1, -1], 45.0),
            ([-1, 0], 90.0),
            ([-1, 1], 135.0),
            ([0, 1], 180.0),
            ([1, 1], 225.0),
            ([1, 0], 270.0),
            ([1, -1], 315.0),
        ],
    )
    def test_from_origin(self, current_middle_hip_point, expected):
        before_middle_hip = create_midpoint(np.array([0, 0]))
        current_middle_hip = create_midpoint(np.array(current_middle_hip_point))

        assert np.allclose(get_screen_orientation(before_middle_hip, current_middle_hip), expected)

    @pytest.mark.parametrize(
        "current_middle_hip_point, expected",
        [
            ([1, 0], 0.0),
            ([0, 0], 45.0),
            ([0, 1], 90.0),
            ([0, 2], 135.0),
            ([1, 2], 180.0),
            ([2, 2], 225.0),
            ([2, 1], 270.0),
            ([2, 0], 315.0),
        ],
    )
    def test_from_1_1(self, current_middle_hip_point, expected):
        before_middle_hip = create_midpoint(np.array([1, 1]))
        current_middle_hip = create_midpoint(np.array(current_middle_hip_point))

        assert np.allclose(get_screen_orientation(before_middle_hip, current_middle_hip), expected)


@pytest.mark.parametrize(
    "before_middle_hip_point, before_right_hip_point, expected",
    [
        ([0, 0], [1, 0], 0.0),
        ([0, 0], [1, 1], 45.0),
        ([0, 0], [0, 1], 90.0),
        ([0, 0], [-1, 1], 135.0),
        ([0, 0], [-1, 0], 180.0),
        ([0, 0], [-1, -1], 225.0),
        ([0, 0], [0, -1], 270.0),
        ([0, 0], [1, -1], 315.0),
    ],
)
def test_get_body_orientation(before_middle_hip_point, before_right_hip_point, expected):
    before_middle_hip = create_midpoint(np.array(before_middle_hip_point))
    before_right_hip = Keypoint(xy=np.array(before_right_hip_point), confidence=np.array(1))

    result = get_body_orientation(before_middle_hip, before_right_hip)

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "before_right_point, expected",
    [
        ([-1, 0], 0.0),
        ([-1, -1], 45.0),
        ([0, -1], 90.0),
        ([1, -1], 135.0),
        ([1, 0], 180.0),
        ([1, 1], 225.0),
        ([0, 1], 270.0),
        ([-1, 1], 315.0),
    ],
)
def test_get_moved_degree_origin(before_right_point, expected):
    before_middle_hip = create_midpoint(np.array([0, 0]))
    before_right_hip = Keypoint(xy=np.array(before_right_point), confidence=np.array(1))
    current_middle_hip = create_midpoint(np.array([0, 1]))

    degree = get_moved_degree(before_middle_hip, before_right_hip, current_middle_hip)

    assert np.allclose(degree, expected)


@pytest.mark.parametrize(
    "before_right_point, expected",
    [
        ([0, 0], 0.0),
        ([1, 0], 45.0),
        ([2, 0], 90.0),
        ([2, 1], 135.0),
        ([2, 2], 180.0),
        ([1, 2], 225.0),
        ([0, 2], 270.0),
        ([0, 1], 315.0),
    ],
)
def test_get_moved_degree(before_right_point, expected):
    before_middle_hip = create_midpoint(np.array([1, 1]))
    before_right_hip = Keypoint(xy=np.array(before_right_point), confidence=np.array(1))
    current_middle_hip = create_midpoint(np.array([0, 2]))

    degree = get_moved_degree(before_middle_hip, before_right_hip, current_middle_hip)

    assert np.allclose(degree, expected)


@pytest.mark.parametrize(
    "r, deg, expected",
    [
        (1.0, 0.0, [0, 1]),
        (1.0, 90.0, [-1, 0]),
        (1.0, 180.0, [0, -1]),
        (1.0, 270.0, [1, 0]),
        (1.0, 360.0, [0, 1]),
        (2.0, 0.0, [0, 2]),
        (2.0, 90.0, [-2, 0]),
        (2.0, 180.0, [0, -2]),
        (2.0, 270.0, [2, 0]),
        (2.0, 360.0, [0, 2]),
    ],
)
def test_polar_to_xy(r, deg, expected):
    assert np.allclose(polar_to_xy(r, deg), np.array(expected))
