import numpy as np
import os
from parts.bvh_hierarchy import BVH_HIERARCHY

def convert_3d_to_bvh(filepath, output_dir):
    keypoints_3d_path = os.path.join(output_dir, filepath)
    keypoints_3d_all = np.load(keypoints_3d_path)
    num_frames, num_joints, _ = keypoints_3d_all.shape

    # Create a BVH file for person
    bvh_file_path = os.path.join(output_dir, f'animation_person_{filepath}.bvh')
    with open(bvh_file_path, 'w') as f:
        f.write(BVH_HIERARCHY.format(num_frames=num_frames))
        for frame in range(num_frames):
            for joint in range(num_joints):
                x, y, z = keypoints_3d_all[frame, joint]
                f.write(f"{x} {y} {z} 0 0 0 ")  # Add rotations if needed
            f.write("\n")
    print(f"BVH file for person {keypoints_3d_path} saved to {bvh_file_path}")

if __name__ == "__main__":
    output_dir = 'output_data'
    filepath = 'keypoints_3d.npy'
    convert_3d_to_bvh(filepath, output_dir)