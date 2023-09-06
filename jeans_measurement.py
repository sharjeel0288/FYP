import cv2
import numpy as np
import uuid


def sort_contours_by_area(contours):
    contour_sizes = [(i, cv2.contourArea(cnt))
                     for i, cnt in enumerate(contours)]
    contour_sizes.sort(key=lambda x: x[1], reverse=True)
    return contour_sizes


def get_contours_with_mutation_to_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 0, 150)
    dilated = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_leftmost_contour(contours):
    leftmost_x = float('inf')
    leftmost_index = -1

    for i, contour in enumerate(contours):
        x, _, _, _ = cv2.boundingRect(contour)
        if x < leftmost_x:
            leftmost_index = i
            leftmost_x = x

    return cv2.boundingRect(contours[leftmost_index])


def close_to_one(a):
    return 1 - 1e-3 <= a <= 1 + 1e-3


def get_ppm(ref_rect, real_width, real_height):
    width, height = ref_rect[2], ref_rect[3]
    x = width / real_width
    y = height / real_height
    # if close_to_one(x) and close_to_one(y):
    return (x + y) / 2

    # raise ValueError('Image is not a 90-degree view')


def get_jeans_measurements(points, ppm):
    top = min(points, key=lambda p: p[1])[1]
    bottom = max(points, key=lambda p: p[1])[1]
    left = min(points, key=lambda p: p[0])[0]
    right = max(points, key=lambda p: p[0])[0]

    upper_part = [p for p in points if p[1] - top < (bottom - top) / 4]
    top_left = min(upper_part, key=lambda p: p[0])
    top_right = max(upper_part, key=lambda p: p[0])
    waist = np.sqrt((top_right[0] - top_left[0]) **
                    2 + (top_right[1] - top_left[1])**2) / ppm

    mid_vert_line = (right + left) / 2
    middle_point = min(points, key=lambda p: (p[0] - mid_vert_line)**2)
    rise = distance_point_segment(middle_point, top_left, top_right) / ppm

    lower_right_part = [p for p in points if p[1] - top >
                        (bottom - top) / 4 * 3 and p[0] > mid_vert_line]
    lower_left_part = [p for p in points if p[1] - top >
                       (bottom - top) / 4 * 3 and p[0] < mid_vert_line]

    innermost_left = max(lower_left_part, key=lambda p: p[0])
    innermost_right = min(lower_right_part, key=lambda p: p[0])
    outermost_left = min(lower_left_part, key=lambda p: p[0])
    outermost_right = max(lower_right_part, key=lambda p: p[0])

    inseam_left = distance_point_segment(
        middle_point, innermost_left, innermost_right)
    inseam_right = distance_point_segment(
        middle_point, innermost_right, outermost_right)
    inseam = max(inseam_left, inseam_right) / ppm

    full_leg_left = np.sqrt(
        (top_left[0] - outermost_left[0])**2 + (top_left[1] - outermost_left[1])**2)
    full_leg_right = np.sqrt(
        (top_right[0] - outermost_right[0])**2 + (top_right[1] - outermost_right[1])**2)
    full_leg = max(full_leg_left, full_leg_right) / ppm

    return {
        "waist": waist,
        "rise": rise,
        "inseam": inseam,
        "fullLeg": full_leg
    }


def distance_point_segment(p, a, b):
    t = ((p[0] - a[0]) * (a[0] - b[0]) + (p[1] - a[1]) *
         (a[1] - b[1])) / ((a[0] - b[0])**2 + (a[1] - b[1])**2)
    v = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
    return np.sqrt((v[0] - p[0])**2 + (v[1] - p[1])**2)


def create_contours_stream(image_url, number_of_contours, ref_width, ref_height):
    id = str(uuid.uuid4())
    image = cv2.imread(image_url)
    width, height, _ = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 0, 150)
    dilated = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ref = get_leftmost_contour(contours)
    ppm = get_ppm(ref, ref_width, ref_height)

    sorted_contours = sort_contours_by_area(contours)
    biggest = sorted_contours[:number_of_contours]

    measurements = []

    for i, contour in biggest:
        points = []
        arc_length = cv2.arcLength(contours[i], True)
        epsilon = 0.0025 * arc_length
        approximated_contour = cv2.approxPolyDP(contours[i], epsilon, True)

        cv2.drawContours(image, [contours[i]], 0, (0, 255, 0), 1)
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 1)

        for j in range(contours[i].shape[0]):
            points.append(tuple(contours[i][j][0]))

        measurements.append(get_jeans_measurements(points, ppm))

    # Draw and label measurements
    for measurement in measurements:
        cv2.putText(
            image,
            f"Waist: {measurement['waist']:.2f} in",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            image,
            f"Rise: {measurement['rise']:.2f} in",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            image,
            f"Inseam: {measurement['inseam']:.2f} in",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            image,
            f"Full Leg: {measurement['fullLeg']:.2f} in",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    cv2.imwrite(f"./outputs/{id}.contours.jpg", image)
    cv2.imshow("image", image)
    cv2.waitKey()

    return {
        "uri": f"./outputs/{id}.contours.jpg",
        "id": id,
        "total": len(contours),
        "items": measurements,
    }


# Usage example
image_url = "1809.jpg"  # Replace with your image URL or file path
number_of_contours = 1
ref_width = 0.5
ref_height = 0.5

result = create_contours_stream(
    image_url, number_of_contours, ref_width, ref_height)
print(result)


# "contours": <Number (default=1): number of items you want to get measurements, ordered by size>,
# "refWidth": <Number (default=1): the width of reference object in your preferred length unit>,
# "refHeight": <Number (default=1): the height of reference object in your preferred length unit>
