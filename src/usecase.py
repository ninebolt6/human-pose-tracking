from dataclasses import dataclass

import numpy as np
from calc import mid, warp
from dataclass import Person
from keypoint import KeypointEnum


@dataclass
class AnalysisTarget:
    left_hip: np.ndarray
    right_hip: np.ndarray


@dataclass
class WarpedAnalysisTarget:
    left_hip: np.ndarray
    right_hip: np.ndarray
    mid_point: np.ndarray

    def __init__(self, analysis_target: AnalysisTarget):
        self.left_hip = warp(analysis_target.left_hip)
        self.right_hip = warp(analysis_target.right_hip)
        self.mid_point = mid(self.left_hip, self.right_hip)


def extract_points(person: Person) -> AnalysisTarget:
    left_hip = person.keypoints[KeypointEnum.LEFT_HIP].xy.cpu().numpy()
    right_hip = person.keypoints[KeypointEnum.RIGHT_HIP].xy.cpu().numpy()

    return AnalysisTarget(left_hip, right_hip)
