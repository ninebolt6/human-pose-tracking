import csv
from datetime import datetime
from itertools import chain
import json
import os
from typing import cast

from natsort import natsorted
from calc import length
from dataclass import Person
from keypoint import KeypointEnum

from track import OUTPUT_FOLDER
from usecase import Midpoint, get_body_orientation, warp_keypoints
from util import as_person, sliding_window


TARGET_FOLDER = "20230724_234531"
CSV_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, TARGET_FOLDER)
KEYPOINT_JSON_PATH = os.path.join(OUTPUT_FOLDER, TARGET_FOLDER, "keypoints")
EXEC_TIME = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def get_frame_num(filename: str) -> str:
    return filename.replace("frame_", "").replace(".json", "")


def create_position_header(id: int) -> list[str]:
    return [f"id:{id} x", f"id:{id} y"]


def create_distance_degree_header(id: int) -> list[str]:
    return [f"id:{id} dist", f"id:{id} deg"]


def convert():
    if not os.path.isdir(KEYPOINT_JSON_PATH):
        exit(1)

    files = os.listdir(KEYPOINT_JSON_PATH)
    files = list(filter(lambda f: f.endswith(".json"), files))
    files = natsorted(files)

    with open(os.path.join(CSV_OUTPUT_FOLDER, "output_detail.json"), "r", encoding="utf-8") as detail:
        output_detail = json.load(detail)
        max_person_count = output_detail["max_person_count"]

    with open(
        os.path.join(CSV_OUTPUT_FOLDER, f"out_position_{EXEC_TIME}.csv"), "w", encoding="utf-8", newline=""
    ) as position_out, open(
        os.path.join(CSV_OUTPUT_FOLDER, f"out_distance_degree_{EXEC_TIME}.csv"), "w", encoding="utf-8", newline=""
    ) as distance_degree_out:
        # 位置座標
        position_header = ["frame_num"]
        position_header.extend(
            # flatten
            chain.from_iterable(map(lambda id: create_position_header(id), range(1, max_person_count + 1)))
        )
        position_writer = csv.DictWriter(position_out, fieldnames=position_header)
        position_writer.writeheader()

        # 距離・角度
        distance_degree_header = ["frame_num"]
        distance_degree_header.extend(
            # flatten
            chain.from_iterable(map(lambda id: create_distance_degree_header(id), range(1, max_person_count + 1)))
        )
        distance_degree_writer = csv.DictWriter(distance_degree_out, fieldnames=distance_degree_header)
        distance_degree_writer.writeheader()
        # 前のフレームがないので空行を書き込む
        distance_degree_writer.writerow({"frame_num": get_frame_num("1")})

        for window in sliding_window(files, 2):
            (filename, next_filename) = cast(tuple[str, str], window)

            with open(os.path.join(KEYPOINT_JSON_PATH, filename)) as current_file, open(
                os.path.join(KEYPOINT_JSON_PATH, next_filename)
            ) as next_file:
                current_list: list[Person] = json.load(current_file, object_hook=as_person)
                current_person_dict = {person.person_id: person for person in current_list}

                next_list: list[Person] = json.load(next_file, object_hook=as_person)
                next_person_dict = {person.person_id: person for person in next_list}

            # 準備
            current_frame_num = get_frame_num(filename)
            next_frame_num = get_frame_num(next_filename)
            position_dict = {"frame_num": current_frame_num}
            distance_degree_dict = {"frame_num": next_frame_num}

            for person_id in range(1, max_person_count + 1):
                if current_person_dict.get(person_id) is not None:
                    current_warped_keypoints = warp_keypoints(current_person_dict[person_id].keypoints)
                    current_middle_hip = Midpoint(
                        current_warped_keypoints[KeypointEnum.LEFT_HIP],
                        current_warped_keypoints[KeypointEnum.RIGHT_HIP],
                    )

                    # 位置座標
                    current_person_position = current_warped_keypoints[KeypointEnum.LEFT_ANKLE]
                    if not (current_person_position.xy[0] == 0 and current_person_position.xy[1] == 0):
                        position_person_header = create_position_header(person_id)
                        position_dict[position_person_header[0]] = current_person_position.xy[0]
                        position_dict[position_person_header[1]] = current_person_position.xy[1]

                    if next_person_dict.get(person_id) is not None:
                        next_warped_keypoints = warp_keypoints(next_person_dict[person_id].keypoints)
                        next_middle_hip = Midpoint(
                            next_warped_keypoints[KeypointEnum.LEFT_HIP], next_warped_keypoints[KeypointEnum.RIGHT_HIP]
                        )

                        # 距離・角度
                        distance = length(current_middle_hip.xy, next_middle_hip.xy)
                        degree = get_body_orientation(
                            current_middle_hip, next_middle_hip, current_warped_keypoints[KeypointEnum.LEFT_HIP]
                        )

                        distance_degree_person_header = create_distance_degree_header(person_id)
                        distance_degree_dict[distance_degree_person_header[0]] = str(distance)
                        distance_degree_dict[distance_degree_person_header[1]] = str(degree)

            # 書き込み
            position_writer.writerow(position_dict)
            distance_degree_writer.writerow(distance_degree_dict)


if __name__ == "__main__":
    convert()
