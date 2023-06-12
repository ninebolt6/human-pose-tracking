from datetime import datetime
import torch
from ultralytics import YOLO
from ultralytics.yolo.v8.pose.predict import Results
import json
import os
import time

from dataclass import Box, Person, Keypoint
from util import PersonJSONEncoder, parse_result

# main

started_at = time.time()

OUTPUT_FOLDER = "./out"
OUTPUT_NANE = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = os.path.join(OUTPUT_FOLDER, OUTPUT_NANE)

# Load a model
model = YOLO("yolov8n-pose.pt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device, "as device")

results: list[Results] = model.track(
    source="./output.mp4",  # 読み込むファイル
    stream=True,  # メモリを大量に食うのでstreaming処理
    device=device,
    imgsz=1920,
    tracker="./bot-sort.config.yaml",  # 人物追跡のコンフィグ
    save=True,  # 検出結果を動画で保存
    verbose=False,  # ログを抑制
    line_width=2,  # boxの線の太さ
    project=OUTPUT_FOLDER,  # 保存先フォルダ
    name=OUTPUT_NANE  # 保存先サブフォルダ
    # boxes=False,  # 人物の周りに箱を表示するか
    # show_labels=False,  # 人物にラベルを表示するか
)

os.makedirs(os.path.join(OUTPUT_DIR, "keypoints"), exist_ok=True)

for frame_num, result in enumerate(results):
    boxes, keypoints = parse_result(result)
    detected_person_length = boxes.shape[0]

    # print("found", detected_person_length, "person on frame", frame_num)

    data: list[Person] = []

    for i in range(detected_person_length):
        data.append(
            Person(
                id=boxes[i].id.item(),
                box=Box(
                    xyxy=boxes[i].xyxy[0],
                    xyxy_normalized=boxes[i].xyxyn[0],
                    confidence=boxes[i].conf[0],
                ),
                keypoint=Keypoint(
                    points=keypoints[i].xy[0],
                    points_normalized=keypoints[i].xyn[0],
                    has_visible=keypoints[i].has_visible,
                    confidence=keypoints[i].conf[0],
                ),
            )
        )

    with open(os.path.join(OUTPUT_DIR, f"keypoints/frame_{frame_num}.json"), "w") as f:
        json.dump(data, f, cls=PersonJSONEncoder)


ended_at = time.time()

print("実行時間: ", ended_at - started_at, "秒")
