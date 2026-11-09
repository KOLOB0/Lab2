import numpy as np

def projection_error(H12, CL1uv, CL2uv):
    """Given two list of coordinates (CL1uv and CL2uv) on two images and the
    homography that relates them (H12) this function will compute an error vector
    (errorVec).  This vector will contain the Euclidean distance between each
    point in CL1uv and its corresponding point in CL2uv after applying the
    homography  H12.

    Args:
        H12 (numpy.ndarray): Homography relating image #1 and image #2. 3x3 matrix.
        CL1uv (numpy.ndarray): Set of points on image #1. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
        CL2uv (numpy.ndarray): Set of points on image #2. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.

    Returns:
        numpy.ndarray: Set of L2 norm's calculated between the original and projected points. Size: Nx1, with N number of points.
    """

    # Project coordinates of second image onto first one
    #convert the Nx2 (u, v) points to homgenous Nx3 points (u, v, 1)
    CL2uv_homogenous = np.column_stack((CL2uv, np.ones(CL2uv.shape[0])))

    # projecting image 2 points onto image 1
    homogeneous_projection = (H12 @ CL2uv_homogenous.T).T

    # normalizing homogenous coordinates back to pixel coordinates
    pixel_projection = homogeneous_projection[:, :2] / homogeneous_projection[:, 2:3]

    # Calculate the projection error vector
    error_vector = np.linalg.norm(CL1uv - pixel_projection, axis = 1) #axis 1 necessary to specify finding the diff between points
             
    return error_vector


