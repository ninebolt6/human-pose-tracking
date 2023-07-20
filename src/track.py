import cv2
from datetime import datetime
import numpy as np
import torch
from ultralytics import YOLO
import json
import os
from time import time
from tqdm import tqdm

from dataclass import Box, Person, Keypoint
from keypoint import KeypointEnum
from util import PersonJSONEncoder, parse_result
from calc import trans_mat, calc_delta_radian, warp_hip_points, mid


VIDEO_PATH = "output.mp4"
OUTPUT_FOLDER = "out"
OUTPUT_NANE = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = os.path.join(OUTPUT_FOLDER, OUTPUT_NANE)

MODEL_NAME = "yolov8x-pose-p6.pt"


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
    save=True,  # 検出結果を動画で保存
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

    # 変換前4点　左上　右上 左下 右下
    src = [[430, 290], [1600, 300], [0, 900], [1920, 900]]
    # 変換行列
    M = trans_mat(src)

    # 射影変換・透視変換する
    output = cv2.warpPerspective(result.orig_img, M, (1920, 1080))

    for person in data:
        (left_now, right_now) = warp_hip_points(person, M)

        # 腰の2点の中点を求める
        mid_now = mid(left_now, right_now)

        # 腰の2点を結ぶ線を描画
        cv2.line(
            output,
            left_now.astype(int),
            right_now.astype(int),
            (255, 255, 255),
            2,
            cv2.LINE_4,
        )
        # 腰の2点を描画
        cv2.circle(output, left_now.astype(int), 3, (255, 0, 0), -1)
        cv2.circle(output, right_now.astype(int), 3, (0, 255, 0), -1)
        # 腰の中点を描画
        cv2.circle(output, mid_now.astype(int), 3, (0, 0, 255), -1)

        before_person = next(
            filter(lambda p: p.person_id == person.person_id, before_data), None
        )
        if before_person is not None:
            (left_before, right_before) = warp_hip_points(before_person, M)

            mid_before = mid(left_before, right_before)

            LnMn = np.linalg.norm(mid_now - left_now)
            LbMn = np.linalg.norm(mid_now - left_before)
            LnLb = np.linalg.norm(left_before - left_now)

            triangle_theta_1 = np.arccos(
                (LnMn * LnMn + LbMn * LbMn - LnLb * LnLb) / (2 * LnMn * LbMn)
            )

            LbMn = np.linalg.norm(mid_now - left_before)
            MnMb = np.linalg.norm(mid_now - mid_before)
            LbMb = np.linalg.norm(mid_before - left_before)

            triangle_theta_2 = np.arccos(
                (LbMn * LbMn + MnMb * MnMb - LbMb * LbMb) / (2 * LbMn * MnMb)
            )

            theta = triangle_theta_1 + triangle_theta_2

            cv2.putText(
                output,
                f"Person {person.person_id}: {theta:.2f}deg",
                mid_now.astype(int),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            left_before = left_before.astype(int)
            right_before = right_before.astype(int)
            # 腰の2点を結ぶ線を描画
            cv2.line(
                output,
                left_before.astype(int),
                right_before,
                (255, 255, 255),
                2,
                cv2.LINE_4,
            )
            # 腰の2点を描画
            cv2.circle(output, left_before.astype(int), 3, (255, 0, 0), -1)
            cv2.circle(output, right_before.astype(int), 3, (0, 255, 0), -1)

            cv2.circle(output, mid_before.astype(int), 3, (0, 0, 255), -1)
            cv2.line(
                output,
                mid_now.astype(int),
                mid_before.astype(int),
                (255, 0, 255),
                2,
                cv2.LINE_4,
            )

    cv2.imshow("frame", output)
    cv2.waitKey(0)

    before_data = data
    with open(os.path.join(OUTPUT_DIR, f"keypoints/frame_{frame_num}.json"), "w") as f:
        json.dump(data, f, cls=PersonJSONEncoder)

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
