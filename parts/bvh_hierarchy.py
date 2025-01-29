BVH_HIERARCHY_ROT = """
HIERARCHY
ROOT Hips
{{
    OFFSET 0 0 0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {{
        OFFSET 0 1 0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Neck
        {{
            OFFSET 0 1 0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Head
            {{
                OFFSET 0 1 0
                CHANNELS 3 Zrotation Xrotation Yrotation
            }}
        }}
        JOINT LeftShoulder
        {{
            OFFSET -0.5 0 0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftElbow
            {{
                OFFSET -1 0 0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT LeftWrist
                {{
                    OFFSET -1 0 0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                }}
            }}
        }}
        JOINT RightShoulder
        {{
            OFFSET 0.5 0 0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightElbow
            {{
                OFFSET 1 0 0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT RightWrist
                {{
                    OFFSET 1 0 0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                }}
            }}
        }}
    }}
    JOINT LeftHip
    {{
        OFFSET -0.5 -1 0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftKnee
        {{
            OFFSET 0 -1 0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftAnkle
            {{
                OFFSET 0 -1 0
                CHANNELS 3 Zrotation Xrotation Yrotation
            }}
        }}
    }}
    JOINT RightHip
    {{
        OFFSET 0.5 -1 0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightKnee
        {{
            OFFSET 0 -1 0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightAnkle
            {{
                OFFSET 0 -1 0
                CHANNELS 3 Zrotation Xrotation Yrotation
            }}
        }}
    }}
}}
MOTION
Frames: {num_frames}
Frame Time: {frame_time}
"""
