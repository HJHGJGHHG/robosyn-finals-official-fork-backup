"""Dataset schema for task-specific ManipulatePipette ParaPred."""
TASK_NAME = "manipulate_pipette"
CAMERAS = ("cam_high", "cam_right_wrist", "cam_left_wrist")
QPOS_KEYS = ("left_arm_init_qpos", "right_arm_init_qpos")
POSE_KEYS = (
    "pipette_base_pose",
    "beaker1_pose",
    "left_arm_base_pose",
    "right_arm_base_pose",
)
TARGET_KEYS = QPOS_KEYS + POSE_KEYS
