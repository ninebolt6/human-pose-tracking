import cv2
import numpy as np

from dataclass import Person
from keypoint import KeypointEnum


def trans_mat(src) -> np.ndarray:
    # 変換前4点　左上　右上 左下 右下
    src = np.array(src, dtype=np.float32)
    # 変換後の4点
    dst = np.array([[0, 0], [1920, 0], [0, 1080], [1920, 1080]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def warp(source: np.ndarray, trans_mat: np.ndarray) -> np.ndarray:
    A = np.dot(trans_mat, np.concatenate((source, [1]), axis=0))
    x = A[0] / A[2]
    y = A[1] / A[2]

    print(x, y)

    if x < 0 or x > 1920:
        x = 0

    if y < 0 or y > 1080:
        y = 0

    if x == 0 or y == -0:
        x = 0
        y = 0
        print("out of range")

    return np.array([x, y], dtype=np.float64)


def warp_hip_points(
    person: Person,
    trans_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_hip = person.keypoints[KeypointEnum.LEFT_HIP].xy.cpu().numpy()
    right_hip = person.keypoints[KeypointEnum.RIGHT_HIP].xy.cpu().numpy()

    return warp(left_hip, trans_mat), warp(right_hip, trans_mat)


def mid(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return (p1 + p2) / 2


def calc_delta_radian(na: np.ndarray, nb: np.ndarray) -> float:
    dot = np.dot(na, nb)
    norm_a = np.linalg.norm(na)
    norm_b = np.linalg.norm(nb)
    return np.arccos(dot / (norm_a * norm_b))
