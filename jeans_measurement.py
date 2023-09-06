from collections import Counter
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


def analyze_jeans_colors(image, jeans_region):
    # Crop the detected jeans region from the image
    x, y, w, h = cv2.boundingRect(jeans_region)
    jeans_cropped = image[y:y+h, x:x+w]

    # Convert cropped region to RGB
    jeans_rgb = cv2.cvtColor(jeans_cropped, cv2.COLOR_BGR2RGB)

    # Flatten the image to a list of RGB tuples
    jeans_pixels = jeans_rgb.reshape(-1, 3)

    # Calculate the dominant colors and their percentages
    total_pixels = len(jeans_pixels)
    color_counts = Counter(map(tuple, jeans_pixels))
    dominant_colors = color_counts.most_common(5)  # Get the top 5 dominant colors

    color_percentages = [(color, count / total_pixels * 100) for color, count in dominant_colors]

    return color_percentages

def create_contours_stream(image_url, number_of_contours, ref_width, ref_height):
    """
    Create contours stream for analyzing jeans measurements and colors.

    Args:
        image_url (str): URL or file path to the input image.
        number_of_contours (int): Number of contours to analyze.
        ref_width (float): The width of the reference object in your preferred length unit.
        ref_height (float): The height of the reference object in your preferred length unit.

    Returns:
        dict: A dictionary containing measurements and color information.
    """
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
         # Analyze jeans colors
        color_percentages = analyze_jeans_colors(image, contours[i])
        
        # Draw and label jeans colors
        color_y = 150
        for color, percentage in color_percentages:
            r, g, b = color
            cv2.putText(
                image,
                f"Color: ({r},{g},{b}) - {percentage:.2f}%",
                (10, color_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            color_y += 30
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
    "items": {'measurements':measurements,'Colors':color_percentages},
    }


def calculate_color_percentages(image_url, k=3):
    # Load the image
    image = cv2.imread(image_url)

    # Convert the image to RGB format (OpenCV loads as BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flatten the image to a 2D array of pixels
    pixels = image_rgb.reshape((-1, 3))

    # Fit K-Means clustering to the pixel data
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get the cluster assignments for each pixel
    cluster_labels = kmeans.labels_

    # Initialize counters for each color
    total_pixels = pixels.shape[0]
    color_counts = {}

    # Iterate through each cluster
    for cluster_id in range(k):
        # Find pixels assigned to this cluster
        cluster_pixels = pixels[cluster_labels == cluster_id]

        # Calculate the percentage of pixels in this cluster
        percentage = (cluster_pixels.shape[0] / total_pixels) * 100

        # Get the average color of this cluster
        average_color = tuple(map(int, np.mean(cluster_pixels, axis=0)))

        # Store the average color and its percentage
        color_counts[average_color] = percentage

    return color_counts
# rgba(132,150,162,255)


# Usage example
image_url = "1809.jpg"  # Replace with your image URL or file path
number_of_contours = 1
ref_width = 0.5
ref_height = 0.5

result = create_contours_stream(
    image_url, number_of_contours, ref_width, ref_height)
print(result)
color_percentages = calculate_color_percentages(image_url)
for color, percentage in color_percentages.items():
    print(f"Color {color}: {percentage:.2f}%")

# "contours": <Number (default=1): number of items you want to get measurements, ordered by size>,
# "refWidth": <Number (default=1): the width of reference object in your preferred length unit>,
# "refHeight": <Number (default=1): the height of reference object in your preferred length unit>
