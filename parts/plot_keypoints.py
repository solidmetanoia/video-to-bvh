import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def transform_to_visualization_coordinates(keypoints):
    """
    Transform keypoints to match the visualization tool's coordinate system.
    :param keypoints: Input keypoints of shape (num_joints, 3).
    :return: Transformed keypoints.
    """
    # Swap Y and Z axes
    keypoints[:, [1, 2]] = keypoints[:, [2, 1]]
    # Invert the Z axis (if necessary)
    keypoints[:, 2] *= -1
    return keypoints

def plot_3d_keypoints_animation(keypoints_3d, skeleton, interval=100):
    """
    Create an animated 3D plot of keypoints over time.
    :param keypoints_3d: 3D keypoints of shape (num_frames, num_joints, 3).
    :param skeleton: List of joint connections as tuples (parent, child).
    :param interval: Delay between frames in milliseconds.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize scatter plot for keypoints
    scatter = ax.scatter([], [], [], c='r', marker='o')

    # Initialize line plots for skeleton connections
    lines = [ax.plot([], [], [], c='b')[0] for _ in skeleton]

    # Set axis limits
    ax.set_xlim([np.min(keypoints_3d[:, :, 0]), np.max(keypoints_3d[:, :, 0])])
    ax.set_ylim([np.min(keypoints_3d[:, :, 1]), np.max(keypoints_3d[:, :, 1])])
    ax.set_zlim([np.min(keypoints_3d[:, :, 2]), np.max(keypoints_3d[:, :, 2])])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def update(frame):
        """
        Update the plot for each frame.
        :param frame: Current frame index.
        """
        # Update scatter plot
        scatter._offsets3d = (
            keypoints_3d[frame, :, 0],
            keypoints_3d[frame, :, 1],
            keypoints_3d[frame, :, 2]
        )

        # Update skeleton connections
        for i, (parent, child) in enumerate(skeleton):
            if not np.isnan(keypoints_3d[frame, parent]).any() and not np.isnan(keypoints_3d[frame, child]).any():
                lines[i].set_data(
                    [keypoints_3d[frame, parent, 0], keypoints_3d[frame, child, 0]],
                    [keypoints_3d[frame, parent, 1], keypoints_3d[frame, child, 1]]
                )
                lines[i].set_3d_properties(
                    [keypoints_3d[frame, parent, 2], keypoints_3d[frame, child, 2]]
                )
            else:
                lines[i].set_data([], [])
                lines[i].set_3d_properties([])

        return scatter, *lines

    # Create animation
    ani = FuncAnimation(
        fig, update, frames=len(keypoints_3d), interval=interval, blit=True
    )

    # Show animation
    plt.show()

# Human3.6M skeleton connections
skeleton = [
    (0, 1),   # Hip -> Right Hip
    (1, 2),   # Right Hip -> Right Knee
    (2, 3),   # Right Knee -> Right Ankle
    (0, 4),   # Hip -> Left Hip
    (4, 5),   # Left Hip -> Left Knee
    (5, 6),   # Left Knee -> Left Ankle
    (0, 7),   # Hip -> Spine
    (7, 8),   # Spine -> Neck
    (8, 9),   # Neck -> Head
    (8, 10),  # Neck -> Left Shoulder
    (10, 11), # Left Shoulder -> Left Elbow
    (11, 12), # Left Elbow -> Left Wrist
    (8, 13),  # Neck -> Right Shoulder
    (13, 14), # Right Shoulder -> Right Elbow
    (14, 15), # Right Elbow -> Right Wrist
]

# Load keypoints
keypoints_3d = np.load('./output_data/keypoints_3d_person_0.npy')  # Replace with your keypoints file

import smoothers
# keypoints_3d = smoothers.low_pass_filter(keypoints_3d, cutoff=0.75, fs=12)
keypoints_3d = smoothers.savgol_smoothing(keypoints_3d)
# keypoints_3d = smoothers.moving_average_smoothing(keypoints_3d)
# Transform keypoints to visualization coordinates
keypoints_3d = transform_to_visualization_coordinates(keypoints_3d)

# Create and display the animation
plot_3d_keypoints_animation(keypoints_3d, skeleton, interval=100)

# if __name__ == "__main__":
#     # keypoints = "./output_data/keypoints_3d_person_0.npy"
#     keypoints = np.load("./output_data/keypoints_3d_person_0.npy")
#     plot_3d_keypoints(keypoints[0], skeleton)