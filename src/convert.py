import csv
from datetime import datetime
from itertools import chain
import json
import os

from natsort import natsorted
from dataclass import Person
from keypoint import KeypointEnum

from track import OUTPUT_FOLDER
from usecase import warp_keypoints
from util import as_person


TARGET_FOLDER = "20230724_234531"
CSV_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, TARGET_FOLDER)
KEYPOINT_JSON_PATH = os.path.join(OUTPUT_FOLDER, TARGET_FOLDER, "keypoints")
EXEC_TIME = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"


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
    ) as position_out:
        position_header = ["frame_num"]
        position_header.extend(
            # flatten
            chain.from_iterable(map(lambda id: [f"id:{id} x", f"id:{id} y"], range(1, max_person_count + 1)))
        )
        position_writer = csv.DictWriter(position_out, fieldnames=position_header)
        position_writer.writeheader()

        for filename in files:
            with open(os.path.join(KEYPOINT_JSON_PATH, filename)) as f1:
                current_list: list[Person] = json.load(f1, object_hook=as_person)

            row_dict = {"frame_num": filename.replace(".json", "").replace("frame_", "")}

            for person in current_list:
                warped_keypoints = warp_keypoints(person.keypoints)

                # 位置座標
                person_position = warped_keypoints[KeypointEnum.LEFT_ANKLE]

                if person_position.xy[0] == 0 and person_position.xy[1] == 0:
                    continue

                row_dict[f"id:{person.person_id} x"] = person_position.xy[0]
                row_dict[f"id:{person.person_id} y"] = person_position.xy[1]

            position_writer.writerow(row_dict)


if __name__ == "__main__":
    convert()
