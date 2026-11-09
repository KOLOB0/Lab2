import os 
import cv2 
import numpy as np
import matplotlib.pyplot as plt 

from src.internal.match_sift import match_sift, match_sift_opencv
from src.projection_error import projection_error

def Section3():

    print("Running Section 3...")

    img1_name = "imgl01311.jpg"
    img2_name = "imgl01396.jpg"

    #img1_name = "IMG_1253_small.JPG"
    #img2_name = "IMG_1254_small.JPG"

    ############## LOAD DATA ################

    # Load images from disk
    #project_dir = os.getcwd()
    project_dir = os.path.dirname(os.path.abspath(__file__))
    img1_path = os.path.join(project_dir, "MiamiSet00", img1_name)
    img2_path = os.path.join(project_dir, "MiamiSet00", img2_name)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)


    ############## SECTION 3 ################

    H12 = np.array([[0.923963,  0.105144,  -41.64614],
                  [-0.105144, 0.923963, -224.7121],
                  [0.0,       0.0,         1.0]])


 #   H12 = np.matrix([[0.991966, 0.135529, -85.4565],
 #                   [-0.135529, 0.991966, -372.3004],
 #                   [0, 0, 1]])
    


    # Do feature association with the modified match function
    dist_ratio = 0.8
    draw_matches = True
    CL1uv, CL2uv, kpts1, descs1, kpts2, descs2 = match_sift(img1, img2, dist_ratio, draw_matches)

    # This is an alternative way of finding matches, filtering them with Lowe's ratio 
    # and showing the associations, using opencv functions
    matches = match_sift_opencv(descs1, descs2, dist_ratio)
    img3 = cv2.drawMatchesKnn(img1, kpts1, img2, kpts2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    print(CL1uv)
    plt.imshow(img3, origin="upper"); plt.show()

    draw_matches = False
    dist_threshold = 50

    # Initialize stacks for plotting. 
    dist_ratios = []
    avg_rep_errors = []
    max_rep_errors = []
    num_points = []
    num_points_over_threshold = []

    # Iterate over different Lowe's ratio 
    step = 0.05
    for dist_ratio in np.arange(0.4, 0.9 + step, step):

        # Find matches between image #1 and #2 and calculate the projection error 
        CL1uv, CL2uv, kpts1, descs1, kpts2, descs2 = match_sift(img1, img2, dist_ratio, draw_matches)
        error_vec = projection_error(H12,CL1uv,CL2uv)

        if error_vec.size == 0:
            print("No features were associated.")
        else:
            dist_ratios.append(dist_ratio)
            avg_rep_errors.append(np.mean(error_vec))
            max_rep_errors.append(np.max(error_vec))
            num_points.append(CL1uv.shape[0])
            num_points_over_threshold.append(np.sum(error_vec > dist_threshold))

    # Draw two versions of the same plot, one with linear y-axis and the other
    # with log y-axis, to better see the small vs large values
    _, ax = plt.subplots(figsize=(10,5), nrows=1, ncols=2)

    ax[0].plot(dist_ratios, avg_rep_errors, label="Avg Rep Error")
    ax[0].plot(dist_ratios, max_rep_errors, label="Max Rep Error")
    ax[0].plot(dist_ratios, num_points, label="Num Matches")
    ax[0].plot(dist_ratios, num_points_over_threshold, label="Num matches > dist threshold")
    ax[0].set_xlabel("Distance Ratio")
    ax[0].set_ylabel("Error and Num Matches (linear scale)")
    ax[0].legend()

    ax[1].semilogy(dist_ratios, avg_rep_errors, label="Avg Rep Error")
    ax[1].semilogy(dist_ratios, max_rep_errors, label="Max Rep Error")
    ax[1].semilogy(dist_ratios, num_points, label="Num Matches")
    ax[1].semilogy(dist_ratios, num_points_over_threshold, label="Num matches > dist threshold")
    ax[1].set_xlabel("Distance Ratio")
    ax[1].set_ylabel("Error and Num Matches (log scale)")
    ax[1].legend()

    plt.show()

if __name__ == '__main__':
    Section3()
