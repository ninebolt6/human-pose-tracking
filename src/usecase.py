from dataclasses import dataclass

import numpy as np
from calc import mid, warp
from dataclass import Person
from keypoint import KeypointEnum


@dataclass
class WarpedAnalysisTarget:
    left_hip: np.ndarray
    right_hip: np.ndarray
    mid_point: np.ndarray

    def __init__(self, person: Person):
        self.left_hip = warp(person.keypoints[KeypointEnum.LEFT_HIP].xy)
        self.right_hip = warp(person.keypoints[KeypointEnum.RIGHT_HIP].xy)
        self.mid_point = mid(self.left_hip, self.right_hip)
