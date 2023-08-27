import numpy as np
import pytest

from dataclass import Keypoint
from usecase import Midpoint, get_body_orientation


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


params = [
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


@pytest.mark.parametrize("degree, expected", params)
def test_get_body_orientation(degree, expected):
    before_middle_hip = create_midpoint(np.array([0, 0]))
    current_middle_hip = create_midpoint(create_polar_coordinate(degree))

    assert np.allclose(get_body_orientation(before_middle_hip, current_middle_hip), expected)
