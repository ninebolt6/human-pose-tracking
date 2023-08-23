import numpy as np
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


class TestGetBodyOrientation:
    def test_straight(self):
        current_middle_hip = create_midpoint(np.array([0, 0]))
        next_middle_hip = create_midpoint(create_polar_coordinate(90))

        assert np.allclose(get_body_orientation(current_middle_hip, next_middle_hip), 0.0)

    def test_left_forward(self):
        current_middle_hip = create_midpoint(np.array([0, 0]))
        next_middle_hip = create_midpoint(create_polar_coordinate(120))

        assert np.allclose(get_body_orientation(current_middle_hip, next_middle_hip), 30.0)

    def test_left_backward(self):
        current_middle_hip = create_midpoint(np.array([0, 0]))
        next_middle_hip = create_midpoint(create_polar_coordinate(225))

        assert np.allclose(get_body_orientation(current_middle_hip, next_middle_hip), 135.0)

    def test_right_backward(self):
        current_middle_hip = create_midpoint(np.array([0, 0]))
        next_middle_hip = create_midpoint(create_polar_coordinate(300))

        assert np.allclose(get_body_orientation(current_middle_hip, next_middle_hip), 210.0)

    def test_right_forward(self):
        current_middle_hip = create_midpoint(np.array([0, 0]))
        next_middle_hip = create_midpoint(create_polar_coordinate(45))

        assert np.allclose(get_body_orientation(current_middle_hip, next_middle_hip), 315.0)
