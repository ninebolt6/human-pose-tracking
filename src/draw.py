import cv2

from calc import (
    angle,
    to_degree,
    trans_mat,
)
from constant import DESTINATION_SIZE
from dataclass import Person
from keypoint import KeypointEnum
from usecase import Midpoint, warp_keypoints


def warp_perspective(img):
    return cv2.warpPerspective(img, trans_mat(), DESTINATION_SIZE)


def draw_person(person_data: Person, before_person_data: Person | None, output):
    now_warped_analysis_target = warp_keypoints(person_data.keypoints)

    left_now = now_warped_analysis_target[KeypointEnum.LEFT_HIP]
    right_now = now_warped_analysis_target[KeypointEnum.RIGHT_HIP]
    mid_now = Midpoint(left_now, right_now)

    # 腰の2点を結ぶ線を描画
    cv2.line(
        output,
        left_now.xy.astype(int),
        right_now.xy.astype(int),
        (255, 255, 255),
        2,
        cv2.LINE_4,
    )
    # 腰の2点を描画
    cv2.circle(output, left_now.xy.astype(int), 3, (255, 0, 0), -1)
    cv2.circle(output, right_now.xy.astype(int), 3, (0, 255, 0), -1)
    # 腰の中点を描画
    cv2.circle(output, mid_now.xy.astype(int), 3, (0, 0, 255), -1)

    if before_person_data is not None:
        before_warped_analysis_target = warp_keypoints(before_person_data.keypoints)

        left_before = before_warped_analysis_target[KeypointEnum.LEFT_HIP]
        right_before = before_warped_analysis_target[KeypointEnum.RIGHT_HIP]
        mid_before = Midpoint(left_before, right_before)

        theta = 90.0 - to_degree(angle(mid_before.xy, mid_now.xy, left_before.xy))

        cv2.putText(
            output,
            f"Person {person_data.person_id}: {theta:.2f}deg",
            mid_now.xy.astype(int),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 腰の2点を結ぶ線を描画
        cv2.line(
            output,
            left_before.xy.astype(int),
            right_before.xy.astype(int),
            (255, 255, 255),
            2,
            cv2.LINE_4,
        )
        # 腰の2点を描画
        cv2.circle(output, left_before.xy.astype(int), 3, (255, 0, 0), -1)
        cv2.circle(output, right_before.xy.astype(int), 3, (0, 255, 0), -1)

        cv2.circle(output, mid_before.xy.astype(int), 3, (0, 0, 255), -1)
        cv2.line(
            output,
            mid_now.xy.astype(int),
            mid_before.xy.astype(int),
            (255, 0, 255),
            2,
            cv2.LINE_4,
        )
