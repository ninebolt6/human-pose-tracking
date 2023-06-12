import cv2
from datetime import datetime
import torch
from ultralytics import YOLO
import json
import os
from tqdm import tqdm

from dataclass import Box, Person, Keypoint
from keypoint import KeypointEnum
from util import PersonJSONEncoder, parse_result


VIDEO_PATH = "target.mp4"
OUTPUT_FOLDER = "out"
OUTPUT_NANE = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = os.path.join(OUTPUT_FOLDER, OUTPUT_NANE)

MODEL_NAME = "yolov8x-pose-p6.pt"


# main

model = YOLO(MODEL_NAME)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
results = model.track(
    source=VIDEO_PATH,  # 読み込むファイル
    stream=True,  # メモリを大量に食うのでstreaming処理
    device=device,
    imgsz=1920,
    tracker="config/bot-sort.config.yaml",  # 人物追跡のコンフィグ
    save=True,  # 検出結果を動画で保存
    verbose=False,  # ログを抑制
    line_width=2,  # boxの線の太さ
    project=OUTPUT_FOLDER,  # 保存先フォルダ
    name=OUTPUT_NANE  # 保存先サブフォルダ
    # boxes=False,  # 人物の周りに箱を表示するか
    # show_labels=False,  # 人物にラベルを表示するか
)

os.makedirs(os.path.join(OUTPUT_DIR, "keypoints"), exist_ok=True)

# 操作記録を保存
with open(os.path.join(OUTPUT_DIR, "output_detail.json"), "w") as f:
    json.dump({"device": device.type, "model": MODEL_NAME}, f)


for frame_num, result in enumerate(
    tqdm(
        results,
        total=int(cv2.VideoCapture(VIDEO_PATH).get(cv2.CAP_PROP_FRAME_COUNT)),
        unit="frame",
    )
):
    boxes, keypoints = parse_result(result)
    detected_person_length = boxes.shape[0]

    # print("found", detected_person_length, "person on frame", frame_num)

    data: list[Person] = []

    for i in range(detected_person_length):
        keypointDict: dict[KeypointEnum, Keypoint] = {}

        for keypointIndex in KeypointEnum:
            keypointDict[keypointIndex] = Keypoint(
                xy=keypoints[i].xy[0][keypointIndex.value],
                confidence=keypoints[i].conf[0][keypointIndex.value],
            )

        data.append(
            Person(
                person_id=int(boxes[i].id.item()),
                box=Box(
                    xyxy=boxes[i].xyxy[0],
                    confidence=boxes[i].conf[0],
                ),
                keypoints=keypointDict,
            )
        )

    with open(os.path.join(OUTPUT_DIR, f"keypoints/frame_{frame_num}.json"), "w") as f:
        json.dump(data, f, cls=PersonJSONEncoder)
