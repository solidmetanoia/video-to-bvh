import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_rotation(parent_pos, child_pos):
    """
    Compute the rotation (Euler angles) that aligns the parent joint's local Y-axis
    with the direction vector from the parent to the child joint.
    """
    direction = child_pos - parent_pos
    direction /= np.linalg.norm(direction)  # Normalize the direction vector

    # Reshape the direction vector to (1, 3) for align_vectors
    direction = direction.reshape(1, 3)

    # Align the parent's local Y-axis with the direction vector
    rotation = R.align_vectors([[0, 1, 0]], direction)[0]  # Align with Y-axis
    return rotation.as_euler('xyz', degrees=True)  # Convert to Euler angles