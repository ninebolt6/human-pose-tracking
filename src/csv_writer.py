import csv
from io import TextIOWrapper
from itertools import chain
from numpy import float64, ndarray
from usecase import WarpedKeypoint
from abc import ABCMeta, abstractmethod


class CsvWriterBase(metaclass=ABCMeta):
    def __init__(self, file: TextIOWrapper, max_person_count: int):
        header = ["frame_num"]
        header.extend(
            # flatten
            chain.from_iterable(map(lambda id: self.generate_header(id), range(1, max_person_count + 1)))
        )

        self.writer = csv.DictWriter(file, fieldnames=header)
        self.writer.writeheader()
        self.buf = {}

    def append_raw_data(self, key: str, value):
        self.buf[key] = value

    def __clear_buf(self):
        self.buf = {}

    @abstractmethod
    def generate_header(self, id: int) -> list[str]:
        pass

    @abstractmethod
    def append(self):
        pass

    def writerow(self, frame_num: int):
        self.append_raw_data("frame_num", frame_num)
        self.writer.writerow(self.buf)
        self.__clear_buf()


class PositionWriter(CsvWriterBase):
    def generate_header(self, id: int) -> list[str]:
        return [f"id:{id} x", f"id:{id} y"]

    def append(self, person_id: int, position: WarpedKeypoint):
        assert position.xy is not None

        target_header = self.generate_header(person_id)
        self.append_raw_data(target_header[0], position.xy[0])
        self.append_raw_data(target_header[1], position.xy[1])


class DistanceDegreeWriter(CsvWriterBase):
    def generate_header(self, id: int) -> list[str]:
        return [f"id:{id} dist", f"id:{id} deg"]

    def append(self, person_id: int, distance: float64, degree: float64):
        target_header = self.generate_header(person_id)
        self.append_raw_data(target_header[0], distance)
        self.append_raw_data(target_header[1], degree)

    def append_distance(self, person_id: int, distance: float64):
        target_header = self.generate_header(person_id)
        self.append_raw_data(target_header[0], distance)

    def append_degree(self, person_id: int, degree: float64):
        target_header = self.generate_header(person_id)
        self.append_raw_data(target_header[1], degree)


class RelativePositionWriter(CsvWriterBase):
    def generate_header(self, id: int) -> list[str]:
        return [f"id:{id} x", f"id:{id} y"]

    def append(self, person_id: int, relative_position: ndarray):
        target_header = self.generate_header(person_id)
        self.append_raw_data(target_header[0], relative_position[0])
        self.append_raw_data(target_header[1], relative_position[1])
