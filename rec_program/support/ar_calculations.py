import numpy as np
from numba import njit

def calculate_rotmat(xdir, zdir, org):
    """
    this function calculates rotation matrix
    """
    v1 = xdir - org  # v1
    v2 = zdir - org  # v2

    vxnorm = v1 / np.linalg.norm(v1)

    vzcap = v2 - (vxnorm.T @ v2) * vxnorm
    vznorm = vzcap / np.linalg.norm(vzcap)

    vynorm = np.cross(vznorm.T[0], vxnorm.T[0]).reshape(3, 1)
    rotMat = np.hstack((vxnorm, vynorm, vznorm))
    return rotMat


@njit
def calculate_rotmat_from_xyo(xdir, ydir, org):
    """
    this function calculates rotation matrix
    """
    v1 = xdir - org  # v1
    v2 = ydir - org  # v2

    vxnorm = v1 / np.linalg.norm(v1)

    vycap = v2 - np.dot(vxnorm.T , v2) * vxnorm
    vynorm = vycap / np.linalg.norm(vycap)

    vznorm = np.cross(vynorm.T[0], vxnorm.T[0]).reshape(3, 1)
    rotMat = np.hstack((vxnorm, vynorm, vznorm))
    return rotMat
