import matplotlib.pyplot as plt
import cv2
from math import sin, cos, radians

def show_keys(image, keypoints, scale_factor=2):
    """Display an image with SIFT keypoints overlaid at higher resolution.

    Args:
        image (cv2.Mat): Grayscale image.
        keypoints (list[cv2.KeyPoint]): List of keypoints inside image.
        scale_factor (float): Factor by which to upscale the image for visualization.
    """

    # Upscale image for higher resolution display
    highres = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
    highres = cv2.merge([highres, highres, highres])  # Convert to 3-channel grayscale

    # Scale keypoints for the new resolution
    scaled_keypoints = []
    for kp in keypoints:
        # cv2.KeyPoint(x, y, size[, angle[, response[, octave[, class_id]]]])
        scaled_kp = cv2.KeyPoint(
            kp.pt[0] * scale_factor,
            kp.pt[1] * scale_factor,
            kp.size * scale_factor,
            kp.angle,
            kp.response,
            kp.octave,
            kp.class_id
        )
        scaled_keypoints.append(scaled_kp)

    # Draw scaled keypoints
    for keypoint in scaled_keypoints:
        highres = transform_line(highres, keypoint, 0.0, 0.0, 1.0, 0.0)
        highres = transform_line(highres, keypoint, 0.85, 0.1, 1.0, 0.0)
        highres = transform_line(highres, keypoint, 0.85, -0.1, 1.0, 0.0)

    # Display image
    plt.figure(figsize=(10, 10))
    plt.imshow(highres, origin="upper")
    plt.axis("off")
    plt.show()


def transform_line(image, keypoint, x1, y1, x2, y2):
    """Draw a line transformed according to keypoint parameters."""
    length = 0.5 * keypoint.size

    # Convert angle from degrees to radians (OpenCV SIFT angles are in degrees)
    theta = radians(keypoint.angle)
    s, c = sin(theta), cos(theta)

    # Apply rotation, translation, and scaling
    r1 = int(round(keypoint.pt[1] - length * (c * y1 + s * x1)))
    c1 = int(round(keypoint.pt[0] + length * (-s * y1 + c * x1)))
    r2 = int(round(keypoint.pt[1] - length * (c * y2 + s * x2)))
    c2 = int(round(keypoint.pt[0] + length * (-s * y2 + c * x2)))

    # Draw cyan arrow lines
    colour = (0, 255, 255)
    thickness = 1
    return cv2.line(image, (c1, r1), (c2, r2), colour, thickness)
