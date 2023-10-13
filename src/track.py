import json
import os
from datetime import datetime
from time import time

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO

from config import get_track_config
from dataclass import Box, Keypoint, Person
from draw import draw_person, warp_perspective
from keypoint import KeypointEnum
from util import PersonJSONEncoder, parse_result

config = get_track_config()

OUTPUT_NANE = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = os.path.join(config.OutputPath, OUTPUT_NANE)


def track():
    started_at = time()

    model = YOLO(config.ModelName)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = model.track(
        source=config.InputPath,  # 読み込むファイル
        stream=True,  # メモリを大量に食うのでstreaming処理
        device=device,
        imgsz=1920,
        tracker="config/bot-sort.config.yaml",  # 人物追跡のコンフィグ
        save=config.OutputEnabled,  # 検出結果を動画で保存
        verbose=False,  # ログを抑制
        line_width=2,  # boxの線の太さ
        project=OUTPUT_DIR,  # 保存先フォルダ
        name=OUTPUT_NANE,  # 保存先サブフォルダ
        agnostic_nms=False,  # 人物のみなのでオフにしてみる
        # boxes=False,  # 人物の周りに箱を表示するか
        # show_labels=False,  # 人物にラベルを表示するか
        # conf=0.30,  # 信頼度の閾値
        # iou=0.60,  # 重複度の閾値
    )

    if config.OutputEnabled:
        os.makedirs(os.path.join(OUTPUT_DIR, "keypoints"), exist_ok=True)

        # 操作記録を保存
        with open(os.path.join(OUTPUT_DIR, "output_detail.json"), "w") as f:
            json.dump(
                {
                    "device": device.type,
                    "model": config.ModelName,
                    "video_path": config.InputPath,
                    "time_elapsed": 0,
                },
                f,
            )

    before_data: list[Person] = []
    person_id_set = set()

    for frame_num, result in enumerate(
        tqdm(
            results,
            total=int(cv2.VideoCapture(config.InputPath).get(cv2.CAP_PROP_FRAME_COUNT)),
            unit="frame",
        )
    ):
        boxes, keypoints = parse_result(result)
        detected_person_length = boxes.shape[0]

        # print("found", detected_person_length, "person on frame", frame_num)

        person_ids_in_frame = set()
        data: list[Person] = []

        for i in range(detected_person_length):
            keypointDict: dict[KeypointEnum, Keypoint] = {}

            if (
                boxes[i].id is None
                or boxes[i].xyxy[0] is None
                or boxes[i].conf[0] is None
                or keypoints[i].xy[0] is None
                or keypoints[i].conf[0] is None
            ):
                print(f"Frame {frame_num}, detected person {i} is None. Skipping.")
                continue

            for keypointIndex in KeypointEnum:
                keypointDict[keypointIndex] = Keypoint(
                    xy=keypoints[i].xy[0][keypointIndex.value].cpu().numpy(),
                    confidence=keypoints[i].conf[0][keypointIndex.value].cpu().numpy(),
                )

            person = Person(
                person_id=int(boxes[i].id.item()),
                box=Box(
                    xyxy=boxes[i].xyxy[0].cpu().numpy(),
                    confidence=boxes[i].conf[0].cpu().numpy(),
                ),
                keypoints=keypointDict,
            )

            person_ids_in_frame.add(person.person_id)
            data.append(person)

        if config.ShowPreview:
            # 射影変換・透視変換する
            output = warp_perspective(result.orig_img, config.SourcePoints, config.DestinationSize)

            for person in data:
                before_person = next(filter(lambda p: p.person_id == person.person_id, before_data), None)

                draw_person(person, before_person, output)

            cv2.imshow("frame", output)
            cv2.waitKey(0)

        person_id_set.update(person_ids_in_frame)
        before_data = data

        if config.OutputEnabled:
            with open(os.path.join(OUTPUT_DIR, f"keypoints/frame_{frame_num + 1}.json"), "w") as f:
                json.dump(data, f, cls=PersonJSONEncoder)

    if config.OutputEnabled:
        ended_at = time()
        with open(os.path.join(OUTPUT_DIR, "output_detail.json"), "w") as f:
            json.dump(
                {
                    "device": device.type,
                    "model": config.ModelName,
                    "video_path": config.InputPath,
                    "time_elapsed": ended_at - started_at,
                    "max_person_count": max(person_id_set),
                },
                f,
            )


if __name__ == "__main__":
    track()
