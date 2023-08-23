from dataclasses import dataclass

import numpy as np

from calc import angle, to_degree, trans_mat, warp
from config import get_common_config
from dataclass import Keypoint
from keypoint import KeypointEnum

common_config = get_common_config()


@dataclass
class WarpedKeypoint:
    xy: np.ndarray | None
    confidence: np.ndarray

    def __init__(self, keypoint: Keypoint):
        point = warp(keypoint.xy, trans_mat(common_config.SourcePoints, common_config.DestinationSize))

        self.xy = drop_outside(point, common_config.DestinationSize)
        self.confidence = keypoint.confidence


@dataclass
class Midpoint:
    xy: np.ndarray
    confidence: np.ndarray

    def __init__(self, p1: Keypoint | WarpedKeypoint, p2: Keypoint | WarpedKeypoint):
        assert p1.xy is not None and p2.xy is not None

        self.xy = (p1.xy + p2.xy) / 2.0
        self.confidence = p1.confidence * p2.confidence


def warp_keypoints(keypoints: dict[KeypointEnum, Keypoint]) -> dict[KeypointEnum, WarpedKeypoint]:
    return {key: WarpedKeypoint(value) for (key, value) in keypoints.items()}


def get_body_orientation(
    before_middle_hip: Midpoint, current_middle_hip: Midpoint, before_left_hip: Keypoint | WarpedKeypoint
) -> np.float64:
    assert before_left_hip.xy is not None

    return 90 - to_degree(angle(before_middle_hip.xy, current_middle_hip.xy, before_left_hip.xy))


def get_middle_hip(keypoints: dict[KeypointEnum, WarpedKeypoint]) -> Midpoint:
    return Midpoint(keypoints[KeypointEnum.LEFT_HIP], keypoints[KeypointEnum.RIGHT_HIP])


def is_both_hip_exist(keypoints: dict[KeypointEnum, WarpedKeypoint]) -> bool:
    return keypoints[KeypointEnum.LEFT_HIP].xy is not None and keypoints[KeypointEnum.RIGHT_HIP].xy is not None


def drop_outside(xy: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    if xy[0] < 0 or xy[1] < 0 or xy[0] > size[0] or xy[1] > size[1]:
        return None
    return xy
