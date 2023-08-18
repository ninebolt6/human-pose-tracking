import cv2
import numpy as np

from constant import DESTINATION_SIZE, SRC


def trans_mat() -> np.ndarray:
    # 変換前4点　左上　右上 左下 右下
    src = np.array(SRC, dtype=np.float32)
    # 変換後の4点　左上　右上 左下 右下
    dst = np.array(
        [
            [0, 0],
            [DESTINATION_SIZE[0], 0],
            [0, DESTINATION_SIZE[1]],
            DESTINATION_SIZE,
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def warp(source: np.ndarray) -> np.ndarray:
    A = np.dot(trans_mat(), np.concatenate((source, [1]), axis=0))
    x = A[0] / A[2]
    y = A[1] / A[2]

    if x < 0 or x > DESTINATION_SIZE[0]:
        x = 0

    if y < 0 or y > DESTINATION_SIZE[1]:
        y = 0

    if x == 0 or y == 0:
        x = 0
        y = 0
        # TODO: Noneにしたい
        # return None

    return np.array([x, y], dtype=np.float64)


def length(p1: np.ndarray, p2: np.ndarray) -> np.float64:
    return np.linalg.norm(p1 - p2)


def angle(vertex: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.float64:
    """
    三角形の頂点と2点のなす角度を求める

    引数:
        vertex: 頂点
        p1: 1つ目の点
        p2: 2つ目の点

    戻り値:
        2点のなす角度(radian)
    """

    a = length(vertex, p1)
    b = length(vertex, p2)
    c = length(p1, p2)

    return np.arccos((a**2 + b**2 - c**2) / (2 * a * b))


def to_degree(radian: np.float64) -> np.float64:
    return radian * 180 / np.pi
