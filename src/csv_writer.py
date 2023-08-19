import csv
from io import TextIOWrapper
from itertools import chain
from numpy import float64, ndarray
from usecase import WarpedKeypoint


def get_distance_degree_header(id: int) -> list[str]:
    return [f"id:{id} dist", f"id:{id} deg"]


def get_relative_position_header(id: int) -> list[str]:
    return [f"id:{id} x", f"id:{id} y"]


def append_distance_degree(dict: dict[str, str], person_id: int, distance: float64, degree: float64):
    distance_degree_header = get_distance_degree_header(person_id)
    dict[distance_degree_header[0]] = str(distance)
    dict[distance_degree_header[1]] = str(degree)


def append_relative_position(dict: dict[str, str], person_id: int, relative_position: ndarray):
    position_header = get_relative_position_header(person_id)
    dict[position_header[0]] = relative_position[0]
    dict[position_header[1]] = relative_position[1]


class PositionWriter:
    def __init__(self, file: TextIOWrapper, max_person_count: int):
        position_header = ["frame_num"]
        position_header.extend(
            # flatten
            chain.from_iterable(map(lambda id: self.generate_person_header(id), range(1, max_person_count + 1)))
        )

        self.writer = csv.DictWriter(file, fieldnames=position_header)
        self.writer.writeheader()
        self.buf = {}

    def generate_person_header(self, id: int) -> list[str]:
        return [f"id:{id} x", f"id:{id} y"]

    def append_frame_num(self, frame_num: str):
        self.buf["frame_num"] = frame_num

    def append(self, person_id: int, position: WarpedKeypoint):
        assert position.xy is not None

        position_header = self.generate_person_header(person_id)
        self.buf[position_header[0]] = position.xy[0]
        self.buf[position_header[1]] = position.xy[1]

    def writerow(self):
        self.writer.writerow(self.buf)
        self.buf = {}
