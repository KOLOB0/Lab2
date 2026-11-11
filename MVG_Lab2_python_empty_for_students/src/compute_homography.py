import numpy as np
from scipy.linalg import lstsq

def compute_homography(CL1uv, CL2uv, model):
    """Estimate the Homography between two images according to model.

    Args:
        CL1uv (numpy.ndarray): Set of points on image #1. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
        CL2uv (numpy.ndarray): Set of points on image #2. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
        model (str): Type of Homography to estimate. It has to be equal to one of the following strings: 'Translation', 'Similarity', 'Affine', 'Projective'.

    Returns:
        numpy.ndarray: Estimated Homography of Model type. 3x3 matrix.
    """

    # Calculate the matrices for coordinate normalisation  

    # Dummy normalization matrices (identity) - You should replace these with actual normalization
    T1 = np.matrix([[1, 0, 0],
                    [0, 1, 0],
                    [0,0,1]])

    
    T2 = np.matrix([[1, 0, 0],
                    [0, 1, 0],
                    [0,0,1]])
    
    # Normalise the coordinates
    N = CL1uv.shape[0]
    CL1uv_hom = np.hstack([CL1uv, np.ones((N,1))]) 
    CL2uv_hom = np.hstack([CL2uv, np.ones((N,1))]) 
    CL1uv = np.array((T1 @  CL1uv_hom.T ).T)
    CL2uv = np.array((T2 @ CL2uv_hom.T ).T)


    u, v = CL1uv[:, 0], CL1uv[:, 1]
    U, V = CL2uv[:, 0], CL2uv[:, 1]


    if model == "Translation":
        # Create Q matrix 
        Q = np.zeros((2*N,2), dtype=np.float64)
        Q[::2,0] = 1.0    
        Q[1::2,1] = 1.0   

        # Create b vector
        b = np.zeros((2*N), dtype=np.float64)
        b[::2] = CL2uv[:,0] - CL1uv[:,0]   #horizontal movement difference
        b[1::2] = CL2uv[:,1] - CL1uv[:,1]  #vertical movement difference

        # Calculate translation coefficients and build Homography
        x,_,_,_ = lstsq(Q, b)
        (tx, ty) = np.squeeze(x) #removes dimensions of 1
        A = np.eye(3)
        A[:2,2] = [tx, ty]

    elif model == "Similarity":
        # Create Q matrix
        Q = np.zeros((2*N, 4))
        Q[::2, 0] = u
        Q[::2, 1] = -v
        Q[::2, 2] = 1

        Q[1::2, 0] = v
        Q[1::2, 1] = u
        Q[1::2, 3] = 1                  

        # Create b vector
        b = np.zeros(2*N)
        b[::2] = U
        b[1::2] = V       

        # Calculate similarity coefficients and build Homography
        x, _, _, _ = lstsq(Q, b)
        a, beta, tx, ty = x

        A = np.array([
            [a, -beta, tx],
            [beta, a, ty],
            [0, 0, 1] ])


    elif model == "Affine":      
        # Create Q matrix
        Q = np.zeros((2*N, 6))
        Q[::2, 0] = u
        Q[::2, 1] = v
        Q[::2, 2] = 1

        Q[1::2, 3] = u
        Q[1::2, 4] = v
        Q[1::2, 5] = 1

        # Create b vector
        b = np.zeros(2*N)
        b[::2] = U
        b[1::2] = V

         # Calculate affine coefficients and build Homography
        x, _, _, _ = lstsq(Q, b)

        A = np.array([
            [x[0], x[1], x[2]],
            [x[3], x[4], x[5]],
            [0, 0, 1] ])

    elif model == "Projective":
        # Create Q matrix
        Q = np.zeros((2*N, 8))
        Q[::2, 0] = u
        Q[::2, 1] = v
        Q[::2, 2] = 1
        Q[::2, 6] = -U * u
        Q[::2, 7] = -U * v

        Q[1::2, 3] = u
        Q[1::2, 4] = v
        Q[1::2, 5] = 1
        Q[1::2, 6] = -V * u
        Q[1::2, 7] = -V * v 

        # Create b vector
        b = np.zeros(2*N)
        b[::2] = U
        b[1::2] = V
  
        # Calculate projective coefficients and build Homography
        x, _, _, _ = lstsq(Q, b)

        A = np.array([
            [x[0], x[1], x[2]],
            [x[3], x[4], x[5]],
            [x[6], x[7], 1] ])

    else:
        print("Invalid model, returning identity homography")
        H12 = np.eye(3)

    # Un-normalise Homography 
    H12 = np.linalg.inv(T2) @ A @ T1
    return H12