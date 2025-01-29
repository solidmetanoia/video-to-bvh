import numpy as np
import os
import cv2
from mmpose.apis import init_model, inference_topdown
# from mmpose.apis import inference_top_down_pose_model, init_pose_model
from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases
import argparse
from tqdm import tqdm
import pickle

from parts import smoothers

def extract_keypoints(video_path, output_dir, start_time=0, end_time=None, frame_rate=30):
    # inferencer = MMPoseInferencer(pose3d='human')
    inferencer = MMPoseInferencer(pose3d='human3d')
    # inferencer = MMPoseInferencer(pose3d="motionbert_dstformer-ft-243frm_8xb32-120e_h36m")

    # config_file = 'mmpose/configs/body/3d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
    # config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    # # checkpoint_file = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    # checkpoint_file = './td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
    # # checkpoint_file = '.hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    # pose_model = init_model(config_file, checkpoint_file, device='cuda:0')

    # for testing the inferencer
    # for i in inferencer(video_path, show=True, skeleton_style="openpose", device="cuda"):
    #     print(i)
    #     pass
    # exit()

    # for frame-skipping smoothing whatever the fuck
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

        # result_generator = inferencer(frame, vis_out_dir="output_data/test")
        # result_generator = inferencer(frame, show=True)

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
        preproc_keypoints_3d_all_people.append(keypoints_3d_frame)
        # count += 1
        # if count == 4:
        #     break
    cap.release()

    # Save preproc_keypoints_3d_all_people to a pickle file
    pickle_path = os.path.join(output_dir, 'preproc_keypoints_3d_all_people.pkl')
    # with open(pickle_path, 'wb') as f:
    #     pickle.dump(preproc_keypoints_3d_all_people, f)
    # print(f"Preprocessed keypoints saved to {pickle_path}")
    # exit()
    with open(pickle_path, 'rb') as f:
        preproc_keypoints_3d_all_people = pickle.load(f)
        for i, item in enumerate(preproc_keypoints_3d_all_people):
            for j, item2 in enumerate(item[0]):
                preproc_keypoints_3d_all_people = smoothers.low_pass_filter(preproc_keypoints_3d_all_people, window_size=5)
                print(i, j, item2)

        exit()

    # for frame_no, keypoints_3d_frame in enumerate(preproc_keypoints_3d_all_people):
    #     backup_path = os.path.join(output_dir, f'keypoints_3d_frame_{frame_no}.txt')
    #     with open(backup_path, 'w') as f:
    #         for person_no, keypoints_3d in enumerate(keypoints_3d_frame):
    #             for joint in keypoints_3d:
    #                 f.write(f"{person_no} {joint[0]} {joint[1]} {joint[2]}\n")

    num_joints = 17
    keypoints_3d_all_people = np.zeros((len(frame_indices), max_people, num_joints, 3), dtype=np.float32)

    for frame_no, keypoints_3d_frame in enumerate(preproc_keypoints_3d_all_people):
        for person_no, keypoints_3d in enumerate(keypoints_3d_frame):
            keypoints_3d_all_people[frame_no, person_no, :len(keypoints_3d), :] = keypoints_3d

    os.makedirs(output_dir, exist_ok=True)
    output_name = 'keypoints_3d.npy'
    keypoints_3d_path = os.path.join(output_dir, 'keypoints_3d.npy')
    np.save(keypoints_3d_path, np.array(keypoints_3d_all_people))
    print(f"3D keypoints for all people saved to {keypoints_3d_path}")

    for person_no in range(max_people):
        person_keypoints_path = os.path.join(output_dir, f'keypoints_3d_person_{person_no}.npy')
        person_keypoints = keypoints_3d_all_people[:, person_no, :, :]
        np.save(person_keypoints_path, person_keypoints)
        print(f"3D keypoints for person {person_no} saved to {person_keypoints_path}")

    return [output_name]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract 3D keypoints from a video.')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--output_dir', type=str, default='output_data', help='Directory to save the output keypoints')
    parser.add_argument('--start_time', type=float, default=0, help='Start time in seconds')
    parser.add_argument('--end_time', type=float, help='End time in seconds')
    parser.add_argument('--frame_rate', type=int, default=30, help='Frame rate to extract keypoints')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    extract_keypoints(args.video_path, args.output_dir, args.start_time, args.end_time, args.frame_rate)