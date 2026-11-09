import matplotlib.pyplot as plt 
import numpy as np 
import cv2 

def show_warped_images(ax, img1, img2, H12):
    """This function warps img1 using the Homography and then overlays
    img2 and warped images to show the differences.

    Args:
        img1 (cv2.Mat): First grayscale image.
        img2 (cv2.Mat): Second grayscale image.
        H12 (numpy.ndarray): Homography relating image #1 and image #2. 3x3 matrix.
    """

    # Apply Homography to image 
    warped_img = cv2.warpPerspective(img1, H12, img1.shape[::-1])

    # Show overlap with different colours
    overlay = np.zeros((*img1.shape,3))
    overlay[:,:,1] = img2/255.0
    overlay[:,:,0] = warped_img/255.0

    ax.imshow(overlay)
    
