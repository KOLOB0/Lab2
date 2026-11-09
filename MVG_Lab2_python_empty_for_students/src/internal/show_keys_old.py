import matplotlib.pyplot as plt 
import cv2 
from math import sin, cos 

def show_keys(image, keypoints):
    """This function displays an image with SIFT keypoints overlayed.

    Args:
        image (cv2.Mat): Grayscale image.
        keypoints (list[cv2.KeyPoint]): List of keypoints inside image.
    """

    # Create figure 
    plt.figure()

    # create 3-channel grayscale image
    image = cv2.merge([image,image,image])

    # Iterate over every image keypoint
    for keypoint in keypoints:
        # Draw an arrow, each line transformed according to keypoint parameters
        image = transform_line(image, keypoint, 0.0, 0.0, 1.0, 0.0)
        image = transform_line(image, keypoint, 0.85, 0.1, 1.0, 0.0)
        image = transform_line(image, keypoint, 0.85, -0.1, 1.0, 0.0)

    # Show figure
    plt.imshow(image, origin="upper")
    plt.show()


def transform_line(image, keypoint, x1, y1, x2, y2):
    """Draw the given line in the image, but first translate, rotate, and
    scale according to the keypoint parameters.

    Args:
        image (cv2.Mat): RGB image.
        keypoint (_type_): Keypoint storing position, scale and orientation information.
        x1 (float): Beginning of vector x coordinate.
        y1 (float): Beginning of vector y coordinate.
        x2 (float): Ending of vector x coordinate.
        y2 (float): Ending of vector y coordinate.

    Returns:
        cv2.Mat: Image with added line.
    """
    
    # The scaling of the unit length arrow is set to approximately the radius
    # of the region used to compute the keypoint descriptor.
    len = 0.5 * keypoint.size

    # Rotate the keypoints by the orientation
    s = sin(keypoint.angle)
    c = cos(keypoint.angle)

    # Apply transform
    r1 = int(round(keypoint.pt[1] - len * (c * y1 + s * x1)))
    c1 = int(round(keypoint.pt[0] + len * (- s * y1 + c * x1)))
    r2 = int(round(keypoint.pt[1] - len * (c * y2 + s * x2)))
    c2 = int(round(keypoint.pt[0] + len * (- s * y2 + c * x2)))

    # add line to image 
    colour = (0,255,255) # cyan 
    thickness = 1
    return cv2.line(image, (c1,r1), (c2,r2), colour, thickness)


