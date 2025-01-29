# video-to-bvh
## try to read humans in videos to bvh format for basic animations

## EARLY WIP
flow:
- video goes to mmpose
- mmpose processes it into 2d keypoints
- 2d keypoints get estimated into 3d keypoints
- 3d keypoints get processed with IK estimation to blender motion format

### main issues:
- main issue: bvh export motions are weird, visualized keypoints are sometimes off
    - ![image](https://github.com/user-attachments/assets/8f0a7f37-3c4f-4cfe-ac98-eb3d1a788421)
    - left knee is way off.
    - left shoulder too?
    - where the head going

### other issues/thoughts:
- IK estimation is not proper for weird positions
- human3.6 and motionbert can't really capture stuff from behind
  - partial human capture is flimsy. wearing same-color clothes and hiding one body part behind the other too
  - fine if naked.
- maybe I should just do a thing that reads steamvr tracking data into BVH..?
  - would be easier to calculate the 6DOF.
  - I am not pretending to bounce on dick.
- how do I get the foot and hand direction?
- should add hand/face tracking for... motions involved.



```
# if opencv2 has issues with libraries (like I do with ubuntu 22.02), can use agg
MPLBACKEND=agg python3 ./video_to_blender.py ./miku.webm --start_time 49 --end_time 52
MPLBACKEND=agg python3 ./video_to_blender.py ./miku.webm --frame_rate 12
```
