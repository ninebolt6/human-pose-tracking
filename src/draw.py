import cv2

from calc import (
    angle,
    extract_points,
    to_degree,
    trans_mat,
)
from constant import DESTINATION_SIZE


def warp_perspective(img):
    return cv2.warpPerspective(img, trans_mat(), DESTINATION_SIZE)


def draw_person(person_data, before_person_data, output):
    (left_now, right_now, mid_now) = extract_points(person_data)

    # 腰の2点を結ぶ線を描画
    cv2.line(
        output,
        left_now.astype(int),
        right_now.astype(int),
        (255, 255, 255),
        2,
        cv2.LINE_4,
    )
    # 腰の2点を描画
    cv2.circle(output, left_now.astype(int), 3, (255, 0, 0), -1)
    cv2.circle(output, right_now.astype(int), 3, (0, 255, 0), -1)
    # 腰の中点を描画
    cv2.circle(output, mid_now.astype(int), 3, (0, 0, 255), -1)

    if before_person_data is not None:
        (left_before, right_before, mid_before) = extract_points(before_person_data)

        theta = 90.0 - to_degree(angle(mid_before, mid_now, left_before))

        cv2.putText(
            output,
            f"Person {person_data.person_id}: {theta:.2f}deg",
            mid_now.astype(int),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 腰の2点を結ぶ線を描画
        cv2.line(
            output,
            left_before.astype(int),
            right_before,
            (255, 255, 255),
            2,
            cv2.LINE_4,
        )
        # 腰の2点を描画
        cv2.circle(output, left_before.astype(int), 3, (255, 0, 0), -1)
        cv2.circle(output, right_before.astype(int), 3, (0, 255, 0), -1)

        cv2.circle(output, mid_before.astype(int), 3, (0, 0, 255), -1)
        cv2.line(
            output,
            mid_now.astype(int),
            mid_before.astype(int),
            (255, 0, 255),
            2,
            cv2.LINE_4,
        )
