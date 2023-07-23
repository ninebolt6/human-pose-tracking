import cv2

from calc import (
    angle,
    to_degree,
    trans_mat,
)
from constant import DESTINATION_SIZE
from usecase import WarpedAnalysisTarget, extract_points


def warp_perspective(img):
    return cv2.warpPerspective(img, trans_mat(), DESTINATION_SIZE)


def draw_person(person_data, before_person_data, output):
    now_analysis_target = extract_points(person_data)
    now_warped_analysis_target = WarpedAnalysisTarget(now_analysis_target)

    left_now = now_warped_analysis_target.left_hip
    right_now = now_warped_analysis_target.right_hip
    mid_now = now_warped_analysis_target.mid_point

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
        before_analysis_target = extract_points(before_person_data)
        before_warped_analysis_target = WarpedAnalysisTarget(before_analysis_target)

        left_before = before_warped_analysis_target.left_hip
        right_before = before_warped_analysis_target.right_hip
        mid_before = before_warped_analysis_target.mid_point

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
            right_before.astype(int),
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
