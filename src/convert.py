import csv
import json
import os
from datetime import datetime
from itertools import chain

from natsort import natsorted
from numpy import float64, ndarray
from tqdm import tqdm

from calc import length
from config import get_convert_config
from dataclass import Person
from keypoint import KeypointEnum
from usecase import Midpoint, WarpedKeypoint, get_body_orientation, warp_keypoints
from util import as_person

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

        # 相対位置座標
        relative_position_header = ["frame_num"]
        relative_position_header.extend(
            # flatten
            chain.from_iterable(map(lambda id: get_position_header(id), range(1, max_person_count + 1)))
        )
        relative_position_writer = csv.DictWriter(relative_position_out, fieldnames=relative_position_header)
        relative_position_writer.writeheader()

        # key: person_id, value: (key: frame_num, value: person)
        position_cache: dict[int, dict[int, Person]] = {}

        for filename in tqdm(files, unit="frame", total=len(files) - 1):
            with open(os.path.join(KEYPOINT_JSON_PATH, filename)) as current_file:
                current_list: list[Person] = json.load(current_file, object_hook=as_person)
                current_person_dict = {person.person_id: person for person in current_list}

            # 準備
            current_frame_num = get_frame_num(filename)
            position_dict = {"frame_num": current_frame_num}
            distance_degree_dict = {"frame_num": current_frame_num}
            relative_position_dict = {"frame_num": current_frame_num}

            for person_id in range(1, max_person_count + 1):
                if current_person_dict.get(person_id) is not None:
                    current_warped_keypoints = warp_keypoints(current_person_dict[person_id].keypoints)

                    # 位置座標
                    current_person_position = current_warped_keypoints[KeypointEnum.LEFT_ANKLE]
                    if current_person_position.xy is not None:
                        append_position(position_dict, person_id, current_person_position)

                    if (
                        position_cache.get(person_id) is not None
                        and position_cache[person_id].get(int(current_frame_num) - config.CalcInterval) is not None
                    ):
                        before_person = position_cache[person_id][int(current_frame_num) - config.CalcInterval]
                        before_warped_keypoints = warp_keypoints(before_person.keypoints)

                        # 処理する
                        if (
                            before_warped_keypoints[KeypointEnum.LEFT_HIP].xy is not None
                            and before_warped_keypoints[KeypointEnum.RIGHT_HIP].xy is not None
                            and current_warped_keypoints[KeypointEnum.LEFT_HIP].xy is not None
                            and current_warped_keypoints[KeypointEnum.RIGHT_HIP].xy is not None
                        ):
                            # 距離・角度
                            current_middle_hip = Midpoint(
                                current_warped_keypoints[KeypointEnum.LEFT_HIP],
                                current_warped_keypoints[KeypointEnum.RIGHT_HIP],
                            )
                            before_middle_hip = Midpoint(
                                before_warped_keypoints[KeypointEnum.LEFT_HIP],
                                before_warped_keypoints[KeypointEnum.RIGHT_HIP],
                            )
                            distance = length(before_middle_hip.xy, current_middle_hip.xy)
                            degree = get_body_orientation(
                                before_middle_hip, current_middle_hip, current_warped_keypoints[KeypointEnum.LEFT_HIP]
                            )
                            append_distance_degree(distance_degree_dict, person_id, distance, degree)

                        # 相対位置座標
                        before_person_position = before_warped_keypoints[KeypointEnum.LEFT_ANKLE]
                        if before_person_position.xy is not None and current_person_position.xy is not None:
                            relative_position = current_person_position.xy - before_person_position.xy
                            append_relative_position(relative_position_dict, person_id, relative_position)

                        # 書き込めたらキャッシュを削除する
                        del position_cache[person_id]

                    # キャッシュに保存
                    if position_cache.get(person_id) is None:
                        position_cache[person_id] = {}

                    position_cache[person_id][int(current_frame_num)] = current_person_dict[person_id]

            # 書き込み
            position_writer.writerow(position_dict)
            distance_degree_writer.writerow(distance_degree_dict)
            relative_position_writer.writerow(relative_position_dict)


if __name__ == "__main__":
    convert()
    print("Done.")
