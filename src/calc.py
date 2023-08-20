import cv2
import numpy as np


def trans_mat(source, destination_size) -> np.ndarray:
    # 変換前4点　左上　右上 左下 右下
    src = np.array(source, dtype=np.float32)
    # 変換後の4点　左上　右上 左下 右下
    dst = np.array(
        [
            [0, 0],
            [destination_size[0], 0],
            [0, destination_size[1]],
            destination_size,
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def warp(point: np.ndarray, trans_mat) -> np.ndarray:
    A = np.dot(trans_mat, np.concatenate((point, [1]), axis=0))
    x = A[0] / A[2]
    y = A[1] / A[2]

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
