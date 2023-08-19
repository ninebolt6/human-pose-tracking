from numpy import float64, ndarray
from usecase import WarpedKeypoint


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
