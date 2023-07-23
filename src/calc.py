import cv2
import numpy as np

from dataclass import Person
from keypoint import KeypointEnum

DESTINATION_SIZE = (1000, 563)


def trans_mat(src) -> np.ndarray:
    # 変換前4点　左上　右上 左下 右下
    src = np.array(src, dtype=np.float32)
    # 変換後の4点　左上　右上 左下 右下
    dst = np.array(
        [
            [0, 0],
            [DESTINATION_SIZE[0], 0],
            [0, DESTINATION_SIZE[1]],
            [DESTINATION_SIZE[0], DESTINATION_SIZE[1]],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def warp(source: np.ndarray, trans_mat: np.ndarray) -> np.ndarray:
    A = np.dot(trans_mat, np.concatenate((source, [1]), axis=0))
    x = A[0] / A[2]
    y = A[1] / A[2]

    print(x, y)

    if x < 0 or x > 1000:
        x = 0

    if y < 0 or y > 563:
        y = 0

    if x == 0 or y == 0:
        x = 0
        y = 0
        # TODO: Noneにしたい
        # return None

    return np.array([x, y], dtype=np.float64)


def warp_hip_points(
    person: Person,
    trans_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_hip = person.keypoints[KeypointEnum.LEFT_HIP].xy.cpu().numpy()
    right_hip = person.keypoints[KeypointEnum.RIGHT_HIP].xy.cpu().numpy()

    left_warped = warp(left_hip, trans_mat)
    right_warped = warp(right_hip, trans_mat)

    return left_warped, right_warped


def mid(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return (p1 + p2) / 2


def length(p1: np.ndarray, p2: np.ndarray) -> np.float64:
    return np.linalg.norm(p1 - p2)
