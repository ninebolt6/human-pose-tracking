import numpy as np
import pytest

from dataclass import Keypoint
from usecase import Midpoint, get_body_degree, get_body_orientation


def create_midpoint(cood: np.ndarray) -> Midpoint:
    return Midpoint(
        Keypoint(xy=cood, confidence=np.array(1)),
        Keypoint(xy=cood, confidence=np.array(1)),
    )


# 角度から極座標を求める
def create_polar_coordinate(deg: int) -> np.ndarray:
    return np.array(
        [
            np.cos(np.radians([deg])),
            np.sin(np.radians([deg])),
        ]
    )


test_get_body_orientation_params = [
    # 正面ならば0度
    (90, 0.0),
    # 左前ならば30度
    (120, 30.0),
    # 左ならば90度
    (180, 90.0),
    # 左後ろならば135度
    (225, 135.0),
    # 後ろならば180度
    (270, 180.0),
    # 右後ろならば210度
    (300, 210.0),
    # 右ならば270度
    (360, 270.0),
    # 右前ならば315度
    (45, 315.0),
]


@pytest.mark.parametrize("degree, expected", test_get_body_orientation_params)
def test_get_body_orientation(degree, expected):
    before_middle_hip = create_midpoint(np.array([0, 0]))
    current_middle_hip = create_midpoint(create_polar_coordinate(degree))

    assert np.allclose(get_body_orientation(before_middle_hip, current_middle_hip), expected)


test_get_body_degree_origin_params = [
    ([1, 0], 0.0),
    ([1, 1], 315.0),
    ([0, 1], 270.0),
    ([-1, 1], 225.0),
    ([-1, 0], 180.0),
    ([-1, -1], 135.0),
    ([0, -1], 90.0),
    ([1, -1], 45.0),
]


@pytest.mark.parametrize("before_right_point, expected", test_get_body_degree_origin_params)
def test_get_body_degree_origin(before_right_point, expected):
    before_middle_hip = create_midpoint(np.array([0, 0]))
    before_right_hip = Keypoint(xy=np.array(before_right_point), confidence=np.array(1))
    current_middle_hip = create_midpoint(create_polar_coordinate(90))

    degree = get_body_degree(before_middle_hip, before_right_hip, current_middle_hip)

    assert np.allclose(degree, expected)


test_get_body_degree_params = [
    ([2, 1], 225.0),
    ([2, 2], 270.0),
    ([1, 2], 315.0),
    ([0, 2], 0.0),
    ([0, 1], 45.0),
    ([0, 0], 90.0),
    ([1, 0], 135.0),
    ([2, 0], 180.0),
]


@pytest.mark.parametrize("before_right_point, expected", test_get_body_degree_params)
def test_get_body_degree(before_right_point, expected):
    before_middle_hip = create_midpoint(np.array([1, 1]))
    before_right_hip = Keypoint(xy=np.array(before_right_point), confidence=np.array(1))
    current_middle_hip = create_midpoint(create_polar_coordinate(315) + np.array([1, 1]))

    degree = get_body_degree(before_middle_hip, before_right_hip, current_middle_hip)

    assert np.allclose(degree, expected)
