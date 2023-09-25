import json
import os
from datetime import datetime

from natsort import natsorted
from tqdm import tqdm

from calc import length
from config import get_convert_config
from csv_writer import DistanceDegreeWriter, PositionWriter, RelativePositionWriter
from dataclass import Keypoint, Person
from keypoint import KeypointEnum
from position_cache import CacheManager
from usecase import (
    Midpoint,
    WarpedKeypoint,
    get_moved_degree,
    get_middle_hip,
    is_both_hip_exist,
    warp_keypoints,
)
from util import as_person

# 設定の読み込み
config = get_convert_config()
CSV_OUTPUT_FOLDER = os.path.join(config.OutputPath, config.InputPath)
KEYPOINT_JSON_PATH = os.path.join(config.OutputPath, config.InputPath, "keypoints")
EXEC_TIME = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def get_frame_num(filename: str) -> int:
    return int(filename.replace("frame_", "").replace(".json", ""))


def validate_point(point: Keypoint | WarpedKeypoint | Midpoint) -> bool:
    return point.xy is not None and point.confidence >= config.ConfidenceThreshold  # type: ignore


def convert():
    if not os.path.isdir(KEYPOINT_JSON_PATH):
        print(f"Error: {KEYPOINT_JSON_PATH} is not directory.")
        exit(1)

    files = os.listdir(KEYPOINT_JSON_PATH)
    files = list(filter(lambda f: f.endswith(".json"), files))
    files = natsorted(files)

    # 人数の最大値を取得
    with open(os.path.join(CSV_OUTPUT_FOLDER, "output_detail.json"), "r", encoding="utf-8") as detail:
        output_detail = json.load(detail)
        max_person_count = output_detail["max_person_count"]

    # 書き込み用ファイルの準備
    with open(
        os.path.join(CSV_OUTPUT_FOLDER, f"out_position_{EXEC_TIME}.csv"), "w", encoding="utf-8", newline=""
    ) as position_out, open(
        os.path.join(CSV_OUTPUT_FOLDER, f"out_distance_degree_{EXEC_TIME}.csv"), "w", encoding="utf-8", newline=""
    ) as distance_degree_out, open(
        os.path.join(CSV_OUTPUT_FOLDER, f"out_relative_position_{EXEC_TIME}.csv"), "w", encoding="utf-8", newline=""
    ) as relative_position_out:
        # 準備
        position_writer = PositionWriter(position_out, max_person_count)
        distance_degree_writer = DistanceDegreeWriter(distance_degree_out, max_person_count)
        relative_position_writer = RelativePositionWriter(relative_position_out, max_person_count)
        position_cache = CacheManager()

        for filename in tqdm(files, unit="frame", total=len(files) - 1):
            # フレームファイルの読み込み
            with open(os.path.join(KEYPOINT_JSON_PATH, filename)) as current_file:
                current_list: list[Person] = json.load(current_file, object_hook=as_person)
                current_person_dict = {person.person_id: person for person in current_list}
                current_frame_num = get_frame_num(filename)
                calc_target_frame_num = current_frame_num - config.CalcInterval

            for person_id in range(1, max_person_count + 1):
                if current_person_dict.get(person_id) is not None:
                    current_person = current_person_dict[person_id]
                    current_warped_keypoints = warp_keypoints(current_person.keypoints)

                    current_person_position = current_warped_keypoints[config.PersonPositionPoint]
                    if validate_point(current_person_position):
                        # 位置座標の書き込み
                        position_writer.append(person_id, current_person_position)

                    if position_cache.is_calc_target_exist(person_id, calc_target_frame_num):
                        cache = position_cache.get(person_id)
                        assert cache is not None

                        before_person = cache[calc_target_frame_num]
                        before_warped_keypoints = warp_keypoints(before_person.keypoints)
                        before_person_position = before_warped_keypoints[config.PersonPositionPoint]

                        if validate_point(before_person_position) and validate_point(current_person_position):
                            assert before_person_position.xy is not None and current_person_position.xy is not None
                            # 距離の書き込み
                            distance = length(before_person_position.xy, current_person_position.xy)
                            distance_degree_writer.append_distance(person_id, distance)

                            # 相対位置座標の書き込み
                            relative_position = current_person_position.xy - before_person_position.xy
                            relative_position_writer.append(person_id, relative_position)

                        if is_both_hip_exist(current_warped_keypoints) and is_both_hip_exist(before_warped_keypoints):
                            current_middle_hip = get_middle_hip(current_warped_keypoints)
                            before_middle_hip = get_middle_hip(before_warped_keypoints)
                            before_left_hip = before_warped_keypoints[KeypointEnum.LEFT_HIP]
                            before_right_hip = before_warped_keypoints[KeypointEnum.RIGHT_HIP]
                            assert before_left_hip.xy is not None and before_right_hip.xy is not None

                            if validate_point(current_middle_hip) and validate_point(before_middle_hip):
                                # 角度の書き込み
                                degree = get_moved_degree(before_middle_hip, before_right_hip, current_middle_hip)
                                distance_degree_writer.append_degree(person_id, degree)

                        # 書き込めたらキャッシュを削除する
                        position_cache.remove(person_id)

                    # キャッシュに保存
                    position_cache.add(current_person, current_frame_num)

            # 実際にファイルに書き込む
            position_writer.writerow(current_frame_num)
            distance_degree_writer.writerow(current_frame_num)
            relative_position_writer.writerow(current_frame_num)


if __name__ == "__main__":
    convert()
    print("Done.")
