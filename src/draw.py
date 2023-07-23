import cv2

from calc import DESTINATION_SIZE, angle, mid, to_degree, trans_mat, warp_hip_points

# 変換前4点　左上　右上 左下 右下
SRC = [[411, 387], [1281, 390], [70, 794], [1501, 803]]
# 変換行列
M = trans_mat(SRC)


def warp_perspective(img):
    return cv2.warpPerspective(img, M, DESTINATION_SIZE)


def draw_person(person_data, before_person_data, output):
    (left_now, right_now) = warp_hip_points(person_data, M)

    # 腰の2点の中点を求める
    mid_now = mid(left_now, right_now)

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
        (left_before, right_before) = warp_hip_points(before_person_data, M)

        mid_before = mid(left_before, right_before)

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

        left_before = left_before.astype(int)
        right_before = right_before.astype(int)
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
