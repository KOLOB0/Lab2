import numpy as np 
from math import log 

from src.compute_homography import compute_homography
from src.projection_error import projection_error


def compute_homography_ransac(CL1uv, CL2uv, model):
    """Estimate the Homography between two images according to model using the RANSAC algorithm.

    Args:
        CL1uv (numpy.ndarray): Set of points on image #1. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
        CL2uv (numpy.ndarray): Set of points on image #2. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
        model (str): Type of Homography to estimate. It has to be equal to one of the following strings: 'Translation', 'Similarity', 'Affine', 'Projective'.

    Returns:
        numpy.ndarray: Estimated Homography of Model type. 3x3 matrix.
    """
    
    t = 1e1     # RANSAC threshold
    p = 0.98    # probability that at least one random sample is free from outliers

    num_matches = CL1uv.shape[0]   # Number of matching points 

    outlier_percent = 0.5   # Outliers percentage


    if model == "Translation":
        dof = 2
    elif model == "Similarity":
        dof = 4
    elif model == "Affine":
        dof = 6
    elif model == "Projective":
        dof = 8        
    else:
        print("Invalid model")
        return None 
    
    if dof/2 > num_matches:
        print("Not enough matching points..")
        return None 

    #will iterate through randomly chose pairs of idxs of the CL1uv and CL2uv
    #will store best idx pair and update when a new iteration is better
    best_inlier_idxs = None 
    best_consensus_percent = 0.0

    # Calculate the number of iterations according to the current number of estimated outliers and the target outlier percentage.
    num_iter = abs((log(1-p)) / (log(1-(1-outlier_percent)**dof) + 1e-6))

    
    #need to figure number of pairs necessary to compute homography
    sample_size = dof // 2


    rng = np.random.default_rng()
    for _ in range(int(num_iter)):

        
        #choosing random set of pairs to use to compute the homography
        idxs = rng.choice(num_matches, size=sample_size, replace = False)

        # Select a number of random point indices  
        sample1 = CL1uv[idxs]              #randomly sample from CL1uv
        sample2 = CL2uv[idxs]              #randomly sample from CL2uv


        # Estimate the Homography with the selected points
        H = compute_homography(sample1, sample2, model)

        if np.any(np.isnan(H)):
            #If there are problems with H estimation the consensus is valued as null
            
            #consensus_percent = 0
            continue

        else:
            # Compute the consesus related to estimated H
            errors = projection_error(H, CL2uv, CL1uv)
            inliers = errors < t # t is the RANSAC threshold set earlier
            consensus_percent = np.count_nonzero(inliers) / num_matches


            # Update best Homography found
            if consensus_percent > best_consensus_percent:
                best_consensus_percent = consensus_percent
                best_inlier_idxs = np.flatnonzero(inliers)
                

        # Exit condition
        if consensus_percent >= p:
            break

    # Estimate the Homography with the best inliers 
    H = compute_homography(CL1uv[best_inlier_idxs], CL2uv[best_inlier_idxs], model)

    return H