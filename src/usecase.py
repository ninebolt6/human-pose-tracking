from dataclasses import dataclass

import numpy as np

from calc import trans_mat, warp
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


def get_middle_hip(keypoints: dict[KeypointEnum, WarpedKeypoint]) -> Midpoint:
    return Midpoint(keypoints[KeypointEnum.LEFT_HIP], keypoints[KeypointEnum.RIGHT_HIP])


def is_both_hip_exist(keypoints: dict[KeypointEnum, WarpedKeypoint]) -> bool:
    return keypoints[KeypointEnum.LEFT_HIP].xy is not None and keypoints[KeypointEnum.RIGHT_HIP].xy is not None


def drop_outside(xy: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    if xy[0] < 0 or xy[1] < 0 or xy[0] > size[0] or xy[1] > size[1]:
        return None
    return xy


def polar_to_xy(r: np.float64, deg: np.float64) -> np.ndarray:
    # 画面上向きを0度とするため、90度を足す
    rad = np.radians(deg + 90)
    xy = np.array([r * np.cos(rad), r * np.sin(rad)])

    # 画面下向きをy軸の正とするため、y軸を反転させる
    result = reverse_y_axis(xy)
    return result


def normalize_degree(deg) -> np.float64:
    result = deg
    if np.any(result < 0):
        result += 360

    if np.any(result >= 360):
        result -= 360
    return result


def reverse_y_axis(xy: np.ndarray) -> np.ndarray:
    return np.array([xy[0], -xy[1]])


def get_screen_orientation(before_middle_hip: Midpoint, current_middle_hip: Midpoint) -> np.float64:
    # 画面下向きをy軸の正とするため、y軸を反転させる
    before_middle_hip_xy = reverse_y_axis(before_middle_hip.xy)
    current_middle_hip_xy = reverse_y_axis(current_middle_hip.xy)

    # 極座標系で考える
    deg = np.degrees(
        np.arctan2(
            current_middle_hip_xy[1] - before_middle_hip_xy[1],
            current_middle_hip_xy[0] - before_middle_hip_xy[0],
        )
    )

    # 画面の上方向を0度とするため、90度を引く
    result = normalize_degree(deg - 90)
    return result


def get_body_orientation(before_middle_hip: Midpoint, before_right_hip: Keypoint):
    # 極座標系で考える
    deg = np.degrees(
        np.arctan2(before_right_hip.xy[1] - before_middle_hip.xy[1], before_right_hip.xy[0] - before_middle_hip.xy[0])
    )

    result = normalize_degree(deg)
    return result


def get_moved_degree(before_middle_hip, before_right_hip, current_middle_hip):
    screen_orientation = get_screen_orientation(before_middle_hip, current_middle_hip)
    body_orientation = get_body_orientation(before_middle_hip, before_right_hip)

    degree = normalize_degree(screen_orientation + body_orientation)
    return degree
