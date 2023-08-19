import csv
import json
import os
from datetime import datetime
from itertools import chain
from typing import cast

from natsort import natsorted
from numpy import float64, ndarray
from tqdm import tqdm

from calc import length
from config import get_convert_config
from dataclass import Person
from keypoint import KeypointEnum
from usecase import Midpoint, WarpedKeypoint, get_body_orientation, warp_keypoints
from util import as_person, sliding_window

config = get_convert_config()

CSV_OUTPUT_FOLDER = os.path.join(config.OutputPath, config.InputPath)
KEYPOINT_JSON_PATH = os.path.join(config.OutputPath, config.InputPath, "keypoints")
EXEC_TIME = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def get_frame_num(filename: str) -> str:
    return filename.replace("frame_", "").replace(".json", "")


def get_position_header(id: int) -> list[str]:
    return [f"id:{id} x", f"id:{id} y"]


def get_distance_degree_header(id: int) -> list[str]:
    return [f"id:{id} dist", f"id:{id} deg"]


def append_position(dict: dict[str, str], person_id: int, position: WarpedKeypoint):
    assert position.xy is not None

    position_header = get_position_header(person_id)
    dict[position_header[0]] = position.xy[0]
    dict[position_header[1]] = position.xy[1]


def append_distance_degree(dict: dict[str, str], person_id: int, distance: float64, degree: float64):
    distance_degree_header = get_distance_degree_header(person_id)
    dict[distance_degree_header[0]] = str(distance)
    dict[distance_degree_header[1]] = str(degree)


def append_relative_position(dict: dict[str, str], person_id: int, relative_position: ndarray):
    position_header = get_position_header(person_id)
    dict[position_header[0]] = relative_position[0]
    dict[position_header[1]] = relative_position[1]


def convert():
    if not os.path.isdir(KEYPOINT_JSON_PATH):
        print(f"Error: {KEYPOINT_JSON_PATH} is not directory.")
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
    ) as distance_degree_out, open(
        os.path.join(CSV_OUTPUT_FOLDER, f"out_relative_position_{EXEC_TIME}.csv"), "w", encoding="utf-8", newline=""
    ) as relative_position_out:
        # 位置座標
        position_header = ["frame_num"]
        position_header.extend(
            # flatten
            chain.from_iterable(map(lambda id: get_position_header(id), range(1, max_person_count + 1)))
        )
        position_writer = csv.DictWriter(position_out, fieldnames=position_header)
        position_writer.writeheader()

        # 距離・角度
        distance_degree_header = ["frame_num"]
        distance_degree_header.extend(
            # flatten
            chain.from_iterable(map(lambda id: get_distance_degree_header(id), range(1, max_person_count + 1)))
        )
        distance_degree_writer = csv.DictWriter(distance_degree_out, fieldnames=distance_degree_header)
        distance_degree_writer.writeheader()
        # 前のフレームがないので空行を書き込む
        distance_degree_writer.writerow({"frame_num": get_frame_num("1")})

        # 相対位置座標
        relative_position_header = ["frame_num"]
        relative_position_header.extend(
            # flatten
            chain.from_iterable(map(lambda id: get_position_header(id), range(1, max_person_count + 1)))
        )
        relative_position_writer = csv.DictWriter(relative_position_out, fieldnames=relative_position_header)
        relative_position_writer.writeheader()
        # 前のフレームがないので空行を書き込む
        relative_position_writer.writerow({"frame_num": get_frame_num("1")})

        for window in tqdm(sliding_window(files, 2), unit="frame", total=len(files) - 1):
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
            relative_position_dict = {"frame_num": next_frame_num}

            for person_id in range(1, max_person_count + 1):
                if current_person_dict.get(person_id) is not None:
                    current_warped_keypoints = warp_keypoints(current_person_dict[person_id].keypoints)

                    # 位置座標
                    current_person_position = current_warped_keypoints[KeypointEnum.LEFT_ANKLE]
                    if current_person_position.xy is not None:
                        append_position(position_dict, person_id, current_person_position)

                    if next_person_dict.get(person_id) is not None:
                        next_warped_keypoints = warp_keypoints(next_person_dict[person_id].keypoints)

                        if (
                            current_warped_keypoints[KeypointEnum.LEFT_HIP].xy is not None
                            and current_warped_keypoints[KeypointEnum.RIGHT_HIP].xy is not None
                            and next_warped_keypoints[KeypointEnum.LEFT_HIP].xy is not None
                            and next_warped_keypoints[KeypointEnum.RIGHT_HIP].xy is not None
                        ):
                            # 距離・角度
                            next_middle_hip = Midpoint(
                                next_warped_keypoints[KeypointEnum.LEFT_HIP],
                                next_warped_keypoints[KeypointEnum.RIGHT_HIP],
                            )
                            current_middle_hip = Midpoint(
                                current_warped_keypoints[KeypointEnum.LEFT_HIP],
                                current_warped_keypoints[KeypointEnum.RIGHT_HIP],
                            )
                            distance = length(current_middle_hip.xy, next_middle_hip.xy)
                            degree = get_body_orientation(
                                current_middle_hip, next_middle_hip, current_warped_keypoints[KeypointEnum.LEFT_HIP]
                            )
                            append_distance_degree(distance_degree_dict, person_id, distance, degree)

                        # 相対位置座標
                        next_person_position = next_warped_keypoints[KeypointEnum.LEFT_ANKLE]
                        if current_person_position.xy is not None and next_person_position.xy is not None:
                            relative_position = next_person_position.xy - current_person_position.xy
                            append_relative_position(relative_position_dict, person_id, relative_position)

            # 書き込み
            position_writer.writerow(position_dict)
            distance_degree_writer.writerow(distance_degree_dict)
            relative_position_writer.writerow(relative_position_dict)


if __name__ == "__main__":
    convert()
    print("Done.")
