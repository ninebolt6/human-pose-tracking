from json import JSONEncoder
from ultralytics.yolo.v8.pose.predict import Results
from ultralytics.yolo.engine.results import Boxes, Keypoints

from dataclass import Person


class PersonJSONEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Person):
            return o.serialize()
        return super().default(o)


def parse_result(result: Results) -> tuple[Boxes, Keypoints]:
    assert result.boxes is not None
    assert result.keypoints is not None
    assert result.names is not None and result.names[0] == "person"

    boxes = result.boxes
    keypoints = result.keypoints

    return boxes, keypoints