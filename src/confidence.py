import json
import os
from datetime import datetime

from natsort import natsorted
from tqdm import tqdm

from config import get_convert_config
from csv_writer import ConfidenceWriter
from dataclass import Person
from usecase import warp_keypoints
from util import as_person

# 設定の読み込み
config = get_convert_config()
CSV_OUTPUT_FOLDER = os.path.join(config.OutputPath, config.InputPath)
KEYPOINT_JSON_PATH = os.path.join(config.OutputPath, config.InputPath, "keypoints")
EXEC_TIME = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def get_frame_num(filename: str) -> int:
    return int(filename.replace("frame_", "").replace(".json", ""))


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
        os.path.join(CSV_OUTPUT_FOLDER, f"out_confidence_{EXEC_TIME}.csv"), "w", encoding="utf-8", newline=""
    ) as confidence_out:
        # 準備
        confidence_writer = ConfidenceWriter(confidence_out, max_person_count)

        for filename in tqdm(files, unit="frame", total=len(files) - 1):
            # フレームファイルの読み込み
            with open(os.path.join(KEYPOINT_JSON_PATH, filename)) as current_file:
                current_list: list[Person] = json.load(current_file, object_hook=as_person)
                current_person_dict = {person.person_id: person for person in current_list}
                current_frame_num = get_frame_num(filename)

            for person_id in range(1, max_person_count + 1):
                if current_person_dict.get(person_id) is not None:
                    current_person = current_person_dict[person_id]
                    current_warped_keypoints = warp_keypoints(current_person.keypoints)

                    current_person_position = current_warped_keypoints[config.PersonPositionPoint]
                    confidence_writer.append(person_id, current_person_position.confidence)

            # 実際にファイルに書き込む
            confidence_writer.writerow(current_frame_num)


if __name__ == "__main__":
    convert()
    print("Done.")
