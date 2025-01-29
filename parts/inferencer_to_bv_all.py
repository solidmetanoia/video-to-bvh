import numpy as np
import os

from bvh_hierarchy import BVH_HIERARCHY, BVH_HIERARCHY_ROT
from compute_rotations import compute_rotation
import smoothers

joint_hierarchy = {
    "Hips": None,  # Root joint has no parent
    "Spine": "Hips",
    "Neck": "Spine",
    "Head": "Neck",
    "LeftShoulder": "Spine",
    "LeftElbow": "LeftShoulder",
    "LeftWrist": "LeftElbow",
    "RightShoulder": "Spine",
    "RightElbow": "RightShoulder",
    "RightWrist": "RightElbow",
    "LeftHip": "Hips",
    "LeftKnee": "LeftHip",
    "LeftAnkle": "LeftKnee",
    "RightHip": "Hips",
    "RightKnee": "RightHip",
    "RightAnkle": "RightKnee",
}

def convert_3d_to_bvh(keypoints_3d_path, output_dir):
    keypoints_3d_all_people = np.load(keypoints_3d_path)

    num_frames, num_people, num_joints, _ = keypoints_3d_all_people.shape

    # Apply smoothing to the Y values of each joint across all frames
    for person_idx in range(num_people):
        keypoints_3d_all_people[:, person_idx, :] = smoothers.savgol_smoothing(
            keypoints_3d_all_people[:, person_idx, :], window_size=8
        )
        # keypoints_3d_all_people[:, person_idx, :] = smoothers.moving_average_smoothing(
        #     keypoints_3d_all_people[:, person_idx, :], window_size=5
        # )

    # Create a BVH file for each person
    fps = 12
    for person_idx in range(num_people):
        bvh_file_path = os.path.join(output_dir, f'animation_person_{person_idx + 1}.bvh')

        # Smooth the rotations
        rotations = []
        for frame in range(num_frames):
            frame_rotations = []
            for joint in range(1, num_joints):
                parent_joint = keypoints_3d_all_people[frame, person_idx, joint - 1]
                child_joint = keypoints_3d_all_people[frame, person_idx, joint]
                xRot, yRot, zRot = compute_rotation(parent_joint, child_joint)
                frame_rotations.append([xRot, yRot, zRot])
            rotations.append(frame_rotations)

        rotations = np.array(rotations)
        smoothed_rotations = smoothers.savgol_smoothing(rotations, window_size=fps)

        with open(bvh_file_path, 'w') as f:
            f.write(BVH_HIERARCHY_ROT.format(
                num_frames=num_frames,
                frame_time=1 / fps,  # 12 FPS
            ))

            for frame in range(num_frames):
                for joint in range(1, num_joints):
                    x, y, z = keypoints_3d_all_people[frame, person_idx, joint]
                    f.write(f"{x} {y} {z} ")
                    xRot, yRot, zRot = smoothed_rotations[frame, joint - 1]
                    f.write(f"{xRot} {yRot} {zRot} ")
                f.write("\n")

            # for frame in range(num_frames):
            #     for joint in range(1, num_joints):
            #         parent_joint = keypoints_3d_all_people[frame, person_idx, joint - 1]
            #         x, y, z = keypoints_3d_all_people[frame, person_idx, joint]
            #         f.write(f"{x} {y} {z} ")
            #         if joint > 0:
            #             child_joint = keypoints_3d_all_people[frame, person_idx, joint]
            #             xRot, yRot, zRot = compute_rotation(parent_joint, child_joint)
            #             f.write(f"{xRot} {yRot} {zRot} ")
            #         else:
            #             f.write("0 0 0 ")
            #     f.write("\n")
        print(f"BVH file for person {person_idx + 1} saved to {bvh_file_path}")

if __name__ == "__main__":
    output_dir = 'output_data'
    keypoints_3d_path = os.path.join(output_dir, 'keypoints_3d.npy')
    convert_3d_to_bvh(keypoints_3d_path, output_dir)
