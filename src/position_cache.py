from typing import TypeAlias

from dataclass import Person

FrameNum: TypeAlias = int
PersonId: TypeAlias = int


class CacheManager:
    def __init__(self) -> None:
        self.cache: dict[PersonId, dict[FrameNum, Person]] = {}

    def is_calc_target_exist(self, person_id: PersonId, frame_num: FrameNum) -> bool:
        return self.cache.get(person_id) is not None and self.cache[person_id].get(frame_num) is not None

    def get(self, person_id: PersonId) -> dict[FrameNum, Person] | None:
        return self.cache.get(person_id)

    def remove(self, person_id: PersonId):
        del self.cache[person_id]

    def add(self, person: Person, frame_num: FrameNum):
        if self.cache.get(person.person_id) is None:
            self.cache[person.person_id] = {}

        self.cache[person.person_id][frame_num] = person
