import cv2
from datetime import datetime
import torch
from ultralytics import YOLO
import json
import os
from time import time
from tqdm import tqdm

from dataclass import Box, Person, Keypoint
from draw import draw_person, warp_perspective
from keypoint import KeypointEnum
from util import PersonJSONEncoder, parse_result


VIDEO_PATH = "output.mp4"
OUTPUT_FOLDER = "out"
OUTPUT_NANE = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = os.path.join(OUTPUT_FOLDER, OUTPUT_NANE)

MODEL_NAME = "yolov8x-pose-p6.pt"
OUTPUT_ENABLED = True

PREVIEW_MODE = True

# main
started_at = time()

model = YOLO(MODEL_NAME)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
results = model.track(
    source=VIDEO_PATH,  # 読み込むファイル
    stream=True,  # メモリを大量に食うのでstreaming処理
    device=device,
    imgsz=1920,
    tracker="config/bot-sort.config.yaml",  # 人物追跡のコンフィグ
    save=OUTPUT_ENABLED,  # 検出結果を動画で保存
    verbose=False,  # ログを抑制
    line_width=2,  # boxの線の太さ
    project=OUTPUT_FOLDER,  # 保存先フォルダ
    name=OUTPUT_NANE,  # 保存先サブフォルダ
    agnostic_nms=False,  # 人物のみなのでオフにしてみる
    # boxes=False,  # 人物の周りに箱を表示するか
    # show_labels=False,  # 人物にラベルを表示するか
    # conf=0.30,  # 信頼度の閾値
    # iou=0.60,  # 重複度の閾値
)

if OUTPUT_ENABLED:
    os.makedirs(os.path.join(OUTPUT_DIR, "keypoints"), exist_ok=True)

    # 操作記録を保存
    with open(os.path.join(OUTPUT_DIR, "output_detail.json"), "w") as f:
        json.dump(
            {
                "device": device.type,
                "model": MODEL_NAME,
                "video_path": VIDEO_PATH,
                "time_elapsed": 0,
            },
            f,
        )

before_data: list[Person] = []

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

        person = Person(
            person_id=int(boxes[i].id.item()),
            box=Box(
                xyxy=boxes[i].xyxy[0],
                confidence=boxes[i].conf[0],
            ),
            keypoints=keypointDict,
        )

        data.append(person)

    if PREVIEW_MODE:
        # 射影変換・透視変換する
        output = warp_perspective(result.orig_img)

        for person in data:
            before_person = next(
                filter(lambda p: p.person_id == person.person_id, before_data), None
            )

            draw_person(person, before_person, output)

        cv2.imshow("frame", output)
        cv2.waitKey(0)

    before_data = data
    if OUTPUT_ENABLED:
        with open(
            os.path.join(OUTPUT_DIR, f"keypoints/frame_{frame_num}.json"), "w"
        ) as f:
            json.dump(data, f, cls=PersonJSONEncoder)


if OUTPUT_ENABLED:
    ended_at = time()
    with open(os.path.join(OUTPUT_DIR, "output_detail.json"), "w") as f:
        json.dump(
            {
                "device": device.type,
                "model": MODEL_NAME,
                "video_path": VIDEO_PATH,
                "time_elapsed": ended_at - started_at,
            },
            f,
        )
