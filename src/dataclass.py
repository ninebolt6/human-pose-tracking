from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Keypoint:
    # キーポイントの座標
    points: torch.Tensor
    # 縦、横をそれぞれ1としたときのキーポイントの座標
    points_normalized: torch.Tensor
    # キーポイントの確度
    confidence: torch.Tensor
    # キーポイントが見えているかどうか?
    has_visible: bool


@dataclass(frozen=True)
class Box:
    # 左上、右下の座標
    xyxy: torch.Tensor
    # 縦、横をそれぞれ1としたときの位置
    xyxy_normalized: torch.Tensor
    # 確度
    confidence: torch.Tensor


@dataclass(frozen=True)
class Person:
    id: float
    box: Box
    keypoint: Keypoint

    def serialize(self) -> dict:
        obj = {
            "id": self.id,
            "box": {
                "xyxy": self.box.xyxy.tolist(),
                "xyxy_normalized": self.box.xyxy_normalized.tolist(),
                "confidence": self.box.confidence.tolist(),
            },
            "keypoint": {
                "points": self.keypoint.points.tolist(),
                "points_normalized": self.keypoint.points_normalized.tolist(),
                "confidence": self.keypoint.confidence.tolist(),
                "has_visible": self.keypoint.has_visible,
            },
        }
        return obj
