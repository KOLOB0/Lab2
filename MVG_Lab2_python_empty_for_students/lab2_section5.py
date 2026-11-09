import os 
import cv2 
import matplotlib.pyplot as plt 
import scipy 
import numpy as np

from src.internal.match_sift import match_sift, draw_matches_opencv
from src.internal.show_warped_images import show_warped_images
from src.compute_homography_ransac import compute_homography_ransac

def Section5():

    ############## LOAD DATA ################

    # Load features from disk 
    project_dir = os.getcwd()
    features_path = os.path.join(project_dir, "DataSet01", "Features.mat")
    mat = scipy.io.loadmat(features_path)["Features"][0]

    images = np.empty((4,1080,1920))
    features = np.empty((4,64,2))

    # Load images from disk 
    for i in range(4):
        features[i,:,:] = mat[i][0]
        image_file_path = os.path.join(project_dir, "DataSet01", str(i).zfill(2) + ".png")
        images[i,:,:] = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)

    ############## SECTION 5 ################

    # Retrieve the first image and its set of point features 
    img1 = images[0,:,:]
    CL1uv = features[0,:,:]

    for i in range(1, len(features)):

        img2 = images[i,:,:]
        CL2uv = features[i,:,:]

        # Display the associations 
        plt.figure(figsize=(10,5))
        plt.title(f"Matches between images #0 and #{i}")
        draw_matches_opencv(img1, img2, CL1uv, CL2uv)
        plt.show()

        fig, ax = plt.subplots(figsize=(8,8), nrows=2, ncols=2)
        fig.suptitle(f"Homography between images #0 and #{i}")

        for j, model in enumerate(["Translation","Similarity","Affine","Projective"]):
            H12 = compute_homography_ransac(CL1uv, CL2uv, model)
            
            # This is a fuction that warps image I2 into the frame of image I1 and shows the result with red and green colors
            r, c = j//2, j%2
            show_warped_images(ax[r,c], img1, img2, H12)
            ax[r,c].set_title(model)            

        plt.show()

    img1_name = "imgl01311.jpg"
    img2_name = "imgl01396.jpg"

    # Load images from disk
    img1_path = os.path.join(project_dir, "MiamiSet00", img1_name)
    img2_path = os.path.join(project_dir, "MiamiSet00", img2_name)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    dist_ratio = 0.8; 
    CL1uv, CL2uv, _, _, _, _ = match_sift(img1, img2, dist_ratio, False)

    # Display the associations 
    plt.figure(figsize=(10,5))
    plt.title(f"Matches between images {img1_name} and {img2_name}")
    draw_matches_opencv(img1, img2, CL1uv, CL2uv)
    plt.show()

    fig, ax = plt.subplots(figsize=(8,8), nrows=2, ncols=2)
    fig.suptitle(f"Homography between images {img1_name} and {img2_name}")

    for i, model in enumerate(["Translation","Similarity","Affine","Projective"]):

        # Compute Homography matrix 
        H12 = compute_homography_ransac(CL1uv, CL2uv, model)

        # This is a fuction that warps image I2 into the frame of image I1 and shows the result with red and green colors
        r, c = i//2, i%2
        show_warped_images(ax[r,c], img1, img2, H12)
        ax[r,c].set_title(model) 

    plt.show()

if __name__ == '__main__':
    Section5()