import collections
from itertools import islice
from json import JSONEncoder
import numpy as np
from ultralytics.models.yolo.pose.predict import Results
from ultralytics.engine.results import Boxes, Keypoints

from dataclass import Box, Keypoint, Person
from keypoint import KeypointEnum


class PersonJSONEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Person):
            return o.serialize()
        return super().default(o)


def as_person(dct: dict):
    if "person_id" in dct:
        return Person(
            person_id=int(dct["person_id"]),
            box=Box(
                xyxy=np.array(
                    [dct["box"]["top_left_xy"], dct["box"]["bottom_right_xy"]],
                    dtype=np.float64,
                ),
                confidence=np.array(dct["box"]["confidence"], dtype=np.float64),
            ),
            keypoints={
                KeypointEnum[key]: Keypoint(
                    xy=np.array(value["xy"], dtype=np.float64),
                    confidence=np.array(value["confidence"], dtype=np.float64),
                )
                for (key, value) in dct["keypoints"].items()
            },
        )
    return dct


def parse_result(result: Results) -> tuple[Boxes, Keypoints]:
    assert result.boxes is not None
    assert result.keypoints is not None
    assert result.names is not None and result.names[0] == "person"

    boxes = result.boxes
    keypoints = result.keypoints

    return boxes, keypoints


def sliding_window(iterable, n):
    # sliding_window('ABCDEFG', 4) --> ABCD BCDE CDEF DEFG
    it = iter(iterable)
    window = collections.deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)
