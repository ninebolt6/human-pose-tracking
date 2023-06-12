from dataclasses import dataclass
import torch

from keypoint import KeypointEnum


@dataclass(frozen=True)
class Keypoint:
    # キーポイントの座標
    xy: torch.Tensor
    # キーポイントの確度
    confidence: torch.Tensor

    def serialize(self) -> dict:
        return {
            "point": self.xy.tolist(),
            "confidence": self.confidence.item(),
        }


@dataclass(frozen=True)
class Box:
    # 左上、右下の座標
    xyxy: torch.Tensor
    # 確度
    confidence: torch.Tensor


@dataclass(frozen=True)
class Person:
    person_id: int
    box: Box
    keypoints: dict[KeypointEnum, Keypoint]

    def serialize(self) -> dict:
        return {
            "person_id": self.person_id,
            "box": {
                "xyxy": self.box.xyxy.tolist(),
                "confidence": self.box.confidence.tolist(),
            },
            "keypoints": {
                key.name: value.serialize() for (key, value) in self.keypoints.items()
            },
        }
