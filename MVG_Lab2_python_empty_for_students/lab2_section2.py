import cv2 
import os 

from src.internal.show_keys import show_keys

def Section2():
    print("Running Section 2...")

    img1_name = "Img00001_small.JPG"
    img2_name = "Img00025_small.JPG"

    ############## LOAD DATA ################

    # Create SIFT object
    sift = cv2.SIFT_create()

    # Load images from disk
    project_dir = os.path.dirname(os.path.abspath(__file__))
    img1_path = os.path.join(project_dir, "MiamiSet00", img1_name)
    img2_path = os.path.join(project_dir, "MiamiSet00", img2_name)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    ############## SECTION 2 ################

    # Visualise SIFT features for first image 
    kpts1, _ = sift.detectAndCompute(img1, None)
    show_keys(img1, kpts1)

    # Visualise SIFT features for second image 
    kpts2, _ = sift.detectAndCompute(img2, None)
    show_keys(img2, kpts2)

if __name__ == '__main__':
    Section2()
