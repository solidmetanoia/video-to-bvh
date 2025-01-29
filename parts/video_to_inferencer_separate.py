import numpy as np
import os
import cv2
from mmpose.apis import init_model, inference_topdown
# from mmpose.apis import inference_top_down_pose_model, init_pose_model
from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases
import argparse
from tqdm import tqdm

def extract_keypoints(video_path, output_dir, start_time=0, end_time=None, frame_rate=30):
    os.makedirs(output_dir, exist_ok=True)

    # inferencer = MMPoseInferencer(pose3d='human')
    # inferencer = MMPoseInferencer(pose3d='human3d')
    inferencer = MMPoseInferencer(pose3d="motionbert_dstformer-ft-243frm_8xb32-120e_h36m")

    # config_file = 'mmpose/configs/body/3d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
    # config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    # # checkpoint_file = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    # checkpoint_file = './td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
    # # checkpoint_file = '.hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    # pose_model = init_model(config_file, checkpoint_file, device='cuda:0')

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate start and end frame indices
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time else total_frames
    if fps < frame_rate:
        frame_rate = fps
    frame_indices = range(start_frame, end_frame, int(fps / frame_rate))

    preproc_keypoints_3d_all_people = []

    count = 0
    max_people = 0
    for frame_idx in tqdm(frame_indices, desc="Processing frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # pose_results = inference_topdown(pose_model, frame)
        # for result in pose_results:
        #     print(result)

        result_generator = inferencer(frame)
        # result_generator = inferencer(frame, batch_size=5)
        # results = [result for result in result_generator]
        # print(results)
        keypoints_3d_frame = []
        for result in result_generator:
            predictions = result["predictions"][0]
            for prediction in predictions:
                keypoints_3d_frame.append(prediction["keypoints"])
            if max_people < len(predictions):
                max_people = len(predictions)
        # print(keypoints_3d_frame)
        preproc_keypoints_3d_all_people.append(keypoints_3d_frame)
        # count += 1
        # if count == 4:
        #     break
    cap.release()

    num_joints = 17
    keypoints_3d_all_people = {}
    # keypoints_3d_all_people = np.zeros((len(frame_indices), max_people, num_joints, 3), dtype=np.float32)

    for frame_no, keypoints_3d_frame in enumerate(preproc_keypoints_3d_all_people):
        for person_no, keypoints_3d in enumerate(keypoints_3d_frame):
            if person_no not in keypoints_3d_all_people:
                keypoints_3d_all_people[person_no] = []
            keypoints_3d_all_people[person_no].append(keypoints_3d)
            # keypoints_3d_all_people[frame_no, person_no, :len(keypoints_3d), :] = keypoints_3d

    # for i in range(len(preproc_keypoints_3d_all_people)):
    #     print(f"Frame {i}: {preproc_keypoints_3d_all_people[i]}")
    os.makedirs(output_dir, exist_ok=True)
    output_names = []
    for person_no, keypoints_3d_all_people in keypoints_3d_all_people.items():
        output_name = f'keypoints_3d_person_{person_no}.npy'
        output_names.append(output_name)
        keypoints_3d_path = os.path.join(output_dir, output_name)
        np.save(keypoints_3d_path, np.array(keypoints_3d_all_people))
        print(f"3D keypoints for person {person_no} saved to {keypoints_3d_path}")
    # keypoints_3d_path = os.path.join(output_dir, 'keypoints_3d.npy')
    # np.save(keypoints_3d_path, np.array(keypoints_3d_all_people))
    # print(f"3D keypoints for all people saved to {keypoints_3d_path}")
    return output_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract 3D keypoints from a video.')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--output_dir', type=str, default='output_data', help='Directory to save the output keypoints')
    parser.add_argument('--start_time', type=float, default=0, help='Start time in seconds')
    parser.add_argument('--end_time', type=float, help='End time in seconds')
    parser.add_argument('--frame_rate', type=int, default=30, help='Frame rate to extract keypoints')

    args = parser.parse_args()

    extract_keypoints(args.video_path, args.output_dir, args.start_time, args.end_time, args.frame_rate)