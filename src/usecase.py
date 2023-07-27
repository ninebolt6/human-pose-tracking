from dataclasses import dataclass

import numpy as np
from calc import warp
from dataclass import Keypoint
from keypoint import KeypointEnum


@dataclass
class WarpedKeypoint:
    xy: np.ndarray
    confidence: np.ndarray

    def __init__(self, keypoint: Keypoint):
        self.xy = warp(keypoint.xy)
        self.confidence = keypoint.confidence


@dataclass
class Midpoint:
    xy: np.ndarray
    confidence: np.ndarray

    def __init__(self, p1: Keypoint | WarpedKeypoint, p2: Keypoint | WarpedKeypoint):
        self.xy = (p1.xy + p2.xy) / 2.0
        self.confidence = np.min([p1.confidence, p2.confidence])


def warp_keypoints(
    keypoints: dict[KeypointEnum, Keypoint]
) -> dict[KeypointEnum, WarpedKeypoint]:
    return {key: WarpedKeypoint(value) for (key, value) in keypoints.items()}
