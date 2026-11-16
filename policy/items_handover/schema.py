"""Dataset schema for the task-specific ItemsHandover ParaPred policy."""

TASK_NAME = "items_handover"
CAMERAS = ("cam_high", "cam_right_wrist", "cam_left_wrist")
QPOS_KEYS = ("left_arm_init_qpos", "right_arm_init_qpos")
POSE_KEYS = (
    "pen_pose",
    "pen_grasp_pose",
    "holder_pose",
    "left_arm_base_pose",
    "right_arm_base_pose",
)
TARGET_KEYS = QPOS_KEYS + POSE_KEYS
