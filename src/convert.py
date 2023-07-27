import csv
import itertools
import json
import os

from natsort import natsorted
from dataclass import Person
from keypoint import KeypointEnum

from track import OUTPUT_FOLDER
from usecase import Midpoint, warp_keypoints
from util import as_person


TARGET_FOLDER = "20230724_234531"
CSV_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, TARGET_FOLDER)
KEYPOINT_JSON_PATH = os.path.join(OUTPUT_FOLDER, TARGET_FOLDER, "keypoints")


def convert():
    if not os.path.isdir(KEYPOINT_JSON_PATH):
        exit(1)

    with open(
        os.path.join(CSV_OUTPUT_FOLDER, "out.csv"), "w", encoding="utf-8", newline=""
    ) as out, open(
        os.path.join(
            CSV_OUTPUT_FOLDER,
            "output_detail.json",
        ),
        "r",
        encoding="utf-8",
    ) as detail:
        output_detail = json.load(detail)
        max_person_count = output_detail["max_person_count"]

        header = ["frame_num"]
        header.extend(
            # flatten
            itertools.chain.from_iterable(
                map(
                    lambda id: [f"id:{id} x", f"id:{id} y"],
                    range(1, max_person_count + 1),
                )
            )
        )
        writer = csv.DictWriter(out, fieldnames=header)
        writer.writeheader()

        files = os.listdir(KEYPOINT_JSON_PATH)
        files = list(filter(lambda f: f.endswith(".json"), files))
        files = natsorted(files)

        for filename in files:
            with open(os.path.join(KEYPOINT_JSON_PATH, filename)) as f1:
                current_list: list[Person] = json.load(f1, object_hook=as_person)

                row_dict = {
                    "frame_num": filename.replace(".json", "").replace("frame_", "")
                }

                for person in current_list:
                    warped_keypoints = warp_keypoints(person.keypoints)
                    left_point = warped_keypoints[KeypointEnum.LEFT_HIP]
                    right_point = warped_keypoints[KeypointEnum.RIGHT_HIP]
                    mid_point = Midpoint(left_point, right_point)

                    if mid_point.xy[0] == 0 and mid_point.xy[1] == 0:
                        continue

                    row_dict[f"id:{person.person_id} x"] = mid_point.xy[0]
                    row_dict[f"id:{person.person_id} y"] = mid_point.xy[1]

                writer.writerow(row_dict)


if __name__ == "__main__":
    convert()
