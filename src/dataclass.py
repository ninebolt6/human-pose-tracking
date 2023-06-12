from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Keypoint:
    # キーポイントの座標
    points: torch.Tensor
    # キーポイントの確度
    confidence: torch.Tensor


@dataclass(frozen=True)
class Box:
    # 左上、右下の座標
    xyxy: torch.Tensor
    # 確度
    confidence: torch.Tensor


@dataclass(frozen=True)
class Person:
    id: int
    box: Box
    keypoint: Keypoint

    def serialize(self) -> dict:
        obj = {
            "id": self.id,
            "box": {
                "xyxy": self.box.xyxy.tolist(),
                "confidence": self.box.confidence.tolist(),
            },
            "keypoint": {
                "points": self.keypoint.points.tolist(),
                "confidence": self.keypoint.confidence.tolist(),
            },
        }
        return obj
