import cv2 
import matplotlib.pyplot as plt 
import numpy as np

def match_sift(img1, img2, dist_ratio=0.6, draw_matches=True):
    """This function reads two images, finds their SIFT features, and creates
    associations between SIFT features using theri descriptors.
    A match is accepted only if its distance is less than dist_ratio times the distance to the
    second closest match. If draw_matches is true, then it displays lines connecting the matched keypoints.
    It returns two list of matched points, using the uv coordinate convention.
    It also returns the original SIFT locations and descriptors for both images.

    Args:
        img1 (cv2.Mat): First grayscale image.
        img2 (cv2.Mat): Second grayscale image.
        dist_ratio (float, optional): Lowe's ratio. Defaults to 0.6.
        draw_matches (bool, optional): Plotting the matched features or not. Defaults to True.

    Returns:
        (numpy.ndarray, numpy.ndarray, 
        list[cv2.KeyPoint], numpy.ndarray,
        list[cv2.KeyPoint], numpy.ndarray): Tuple containing the set of points on images #1 and #2, 
                                            the keypoints and descriptors in images #1, and   
                                            the keypoints and descriptors in images #2. 
    """

    # Create SIFT object
    sift = cv2.SIFT_create()
    
    # Find SIFT keypoints for each image
    kpts1, descs1 = sift.detectAndCompute(img1, None)
    kpts2, descs2 = sift.detectAndCompute(img2, None)

    colour = (0,255,255) # cyan 
    thickness = 1
    cols1 = img1.shape[1]

    # Convert descriptors to unit vectors
    descs1 = descs1 / np.linalg.norm(descs1, axis=1)[:,None]
    descs2 = descs2 / np.linalg.norm(descs2, axis=1)[:,None]

    # For each descriptor in the first image, select its match to second image.
    descs2t = descs2.T
    match = np.ones((descs1.shape[0]), dtype=np.int64) * -1 
    for i in range(descs1.shape[0]):
        # Computes vector of inverse cosines of dot products 
        angles = np.arccos(descs1[i,:] @ descs2t)   

        # Sort angles and return sorted indices and values
        indx = np.argsort(angles)
        vals = angles[indx]

        # Check if nearest neighbor has angle less than Lowe's ratio times 2nd.
        if (vals[0] < dist_ratio * vals[1]):
            match[i] = indx[0]

    if draw_matches:
        # Create a new image showing the two images side by side
        img3 = cv2.hconcat([img1, img2])

        # create 3-channel grayscale image
        img3 = cv2.merge([img3, img3, img3])

        # Show a figure with lines joining the accepted matches
        for i in range(descs1.shape[0]):
            if match[i] > -1:
                img3 = cv2.line(img3, (int(round(kpts1[i].pt[0])), int(round(kpts1[i].pt[1]))), 
                                      (int(round(kpts2[match[i]].pt[0]) + cols1), int(round(kpts2[match[i]].pt[1]))), 
                                       colour, thickness)
        
        plt.imshow(img3, origin="upper"); plt.show()
        num = np.sum(match > -1)
        print(f"Found {num} matches.")

    # Create the lists of prospective matched points
    num_points = np.sum(match > -1)
    CL1uv = np.zeros((num_points,2))
    CL2uv = np.zeros((num_points,2))
    cl_counter = 0

    for i in range(descs1.shape[0]):
        if (match[i] > -1):
            CL1uv[cl_counter,:] = [int(round(kpts1[i].pt[0])), int(round(kpts1[i].pt[1]))]
            CL2uv[cl_counter,:] = [int(round(kpts2[match[i]].pt[0])), int(round(kpts2[match[i]].pt[1]))]
            cl_counter += 1

    return CL1uv, CL2uv, kpts1, descs1, kpts2, descs2


def match_sift_opencv(descs1, descs2, dist_ratio=0.6):
    """This function finds matches between two sets of descriptors by first running a
    Brute-force matching scheme and then selecting the match if it passes Lowe's ratio test. 
    It returns a list of cv2.DMatch objects containing information about the matching features. 

    Args:
        descs1 (numpy.ndarray): Descriptors in first image. Size: Nx128, with N number of features.
        descs2 (numpy.ndarray): Descriptors in second image. Size: Nx128, with N number of features. 
        dist_ratio (float, optional): Lowe's ratio. Defaults to 0.6.

    Returns:
        list[DMatch]: List of matches (cv2.DMatch) between the two sets of descriptors.  
    """
    
    # Apply Brute Force matching and find two best matches per feature
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descs1, descs2, k=2)

    # Iterate over matches 
    good_matches = []
    for m,n in matches:
        # Apply Lowe's ratio 
        if m.distance < dist_ratio*n.distance:
            good_matches.append([m])

    return good_matches

def draw_matches_opencv(img1, img2, CL1uv, CL2uv):
    """This functions creates cv2.KeyPoint and cv2.DMatch objects from N set of matching points 
    stored inside CL1uv and CL2uv. Afterwards, it uses an OpenCV function to create a combined 
    image of img1 and img2 (side by side) and draws the points and lines to connect the 
    matching points. Finally, it displays the matched features.  

    Args:
        img1 (cv2.Mat): First grayscale image.
        img2 (cv2.Mat): Second grayscale image.
        CL1uv (numpy.ndarray): Set of points on image #1. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
        CL2uv (numpy.ndarray): Set of points on image #2. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
    """

    kpts1 = []; kpts2 = []
    matches = []

    # Iterate over points inside images #1 and #2
    for i, ((x1,y1), (x2,y2)) in enumerate(zip(CL1uv, CL2uv)):
        # Create KeyPoint and DMatch objects
        kpts1.append(cv2.KeyPoint(x1,y1,1))
        kpts2.append(cv2.KeyPoint(x2,y2,1))   
        matches.append(cv2.DMatch(i,i,0.0))

    # Use OpenCV function to draw matches, then display them 
    img3 = cv2.drawMatches(img1.astype(np.uint8), kpts1, img2.astype(np.uint8), kpts2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3, origin="upper")