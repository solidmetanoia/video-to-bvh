import argparse
import os
# from parts.video_to_inferencer_separate import extract_keypoints
from parts.video_to_inferencer_all import extract_keypoints
from parts.inferencer_to_bv_all import convert_3d_to_bvh

parser = argparse.ArgumentParser(description='Extract 3D keypoints from a video.')
parser.add_argument('video_path', type=str, help='Path to the input video file')
parser.add_argument('--output_dir', type=str, default='output_data', help='Directory to save the output keypoints')
parser.add_argument('--start_time', type=float, default=0, help='Start time in seconds')
parser.add_argument('--end_time', type=float, help='End time in seconds')
parser.add_argument('--frame_rate', type=int, default=30, help='Frame rate to extract keypoints')

args = parser.parse_args()

filenames = extract_keypoints(args.video_path, args.output_dir, args.start_time, args.end_time, args.frame_rate)
for filename in filenames:
    keypoints_3d_path = os.path.join(args.output_dir, filename)
    convert_3d_to_bvh(keypoints_3d_path, args.output_dir)