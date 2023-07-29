from dataclasses import dataclass
import numpy as np

from keypoint import KeypointEnum


@dataclass(frozen=True)
class Keypoint:
    # キーポイントの座標
    xy: np.ndarray
    # キーポイントの確度
    confidence: np.ndarray

    def serialize(self) -> dict:
        return {
            "xy": self.xy.tolist(),
            "confidence": self.confidence.item(),
        }


@dataclass(frozen=True)
class Box:
    # 左上、右下の座標
    xyxy: np.ndarray
    # 確度
    confidence: np.ndarray


@dataclass(frozen=True)
class Person:
    person_id: int
    box: Box
    keypoints: dict[KeypointEnum, Keypoint]

    def serialize(self) -> dict:
        return {
            "person_id": self.person_id,
            "box": {
                "top_left_xy": self.box.xyxy[:2].tolist(),
                "bottom_right_xy": self.box.xyxy[2:].tolist(),
                "confidence": self.box.confidence.tolist(),
            },
            "keypoints": {key.name: value.serialize() for (key, value) in self.keypoints.items()},
        }
