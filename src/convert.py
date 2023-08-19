import json
import os
from datetime import datetime

from natsort import natsorted
from tqdm import tqdm

from calc import length
from config import get_convert_config
from csv_writer import (
    PositionWriter,
    DistanceDegreeWriter,
    RelativePositionWriter,
)
from dataclass import Person
from keypoint import KeypointEnum
from usecase import get_body_orientation, get_middle_hip, warp_keypoints
from util import as_person

config = get_convert_config()

CSV_OUTPUT_FOLDER = os.path.join(config.OutputPath, config.InputPath)
KEYPOINT_JSON_PATH = os.path.join(config.OutputPath, config.InputPath, "keypoints")
EXEC_TIME = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def get_frame_num(filename: str) -> str:
    return filename.replace("frame_", "").replace(".json", "")


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
        position_writer = PositionWriter(position_out, max_person_count)

        # 距離・角度
        distance_degree_writer = DistanceDegreeWriter(distance_degree_out, max_person_count)
        # 相対位置座標
        relative_position_writer = RelativePositionWriter(relative_position_out, max_person_count)

        # key: person_id, value: (key: frame_num, value: person)
        position_cache: dict[int, dict[int, Person]] = {}

        for filename in tqdm(files, unit="frame", total=len(files) - 1):
            with open(os.path.join(KEYPOINT_JSON_PATH, filename)) as current_file:
                current_list: list[Person] = json.load(current_file, object_hook=as_person)
                current_person_dict = {person.person_id: person for person in current_list}
                current_frame_num = get_frame_num(filename)

            for person_id in range(1, max_person_count + 1):
                if current_person_dict.get(person_id) is not None:
                    current_warped_keypoints = warp_keypoints(current_person_dict[person_id].keypoints)

                    # 位置座標
                    current_person_position = current_warped_keypoints[KeypointEnum.LEFT_ANKLE]
                    if current_person_position.xy is not None:
                        position_writer.append(person_id, current_person_position)

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
                            current_middle_hip = get_middle_hip(current_warped_keypoints)
                            before_middle_hip = get_middle_hip(before_warped_keypoints)
                            distance = length(before_middle_hip.xy, current_middle_hip.xy)
                            degree = get_body_orientation(
                                before_middle_hip, current_middle_hip, current_warped_keypoints[KeypointEnum.LEFT_HIP]
                            )

                            distance_degree_writer.append(person_id, distance, degree)

                        # 相対位置座標
                        before_person_position = before_warped_keypoints[KeypointEnum.LEFT_ANKLE]
                        if before_person_position.xy is not None and current_person_position.xy is not None:
                            relative_position = current_person_position.xy - before_person_position.xy
                            relative_position_writer.append(person_id, relative_position)

                        # 書き込めたらキャッシュを削除する
                        del position_cache[person_id]

                    # キャッシュに保存
                    if position_cache.get(person_id) is None:
                        position_cache[person_id] = {}

                    position_cache[person_id][int(current_frame_num)] = current_person_dict[person_id]

            # 1行の書き込み
            position_writer.writerow(current_frame_num)
            distance_degree_writer.writerow(current_frame_num)
            relative_position_writer.writerow(current_frame_num)


if __name__ == "__main__":
    convert()
    print("Done.")
