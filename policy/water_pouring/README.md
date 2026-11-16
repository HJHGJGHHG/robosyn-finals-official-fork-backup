# water_pouring

Initial RGB perception and template-based planning for RoboSynChallenge.

- Hugging Face checkpoint: [HJHGJGHHG/robosyn-water_pouring](https://huggingface.co/HJHGJGHHG/robosyn-water_pouring).
- Policy directory: `policy/water_pouring`.
- Deployment configuration: [`deploy_policy.yml`](deploy_policy.yml).
- Required entrypoints: `get_model(usr_args)`, `eval(env, model, observation)`, `reset_model(model)` in `deploy_policy.py`.

From the repository root, after installing the [dependencies](../README.md):

```bash
python policy/download_checkpoints.py --task water_pouring
bash policy/water_pouring/eval.sh 0 random
```

Use a physical GPU index in place of `0`; `random` can be replaced with `clear`. Additional YAML overrides follow the setting, e.g. `--max_episodes 1`. No task-specific CLI overrides are required: all template parameters are already in `deploy_policy.yml`. The checkpoint path is relative to this policy directory: `checkpoint/best.pt`.

RGB inputs are `cam_high`, `cam_right_wrist`, and `cam_left_wrist`, together with public robot joint observations. The policy sends 14-dimensional absolute joint/gripper targets through the official environment interface. The task directory includes its model definition, planner, and static assets.

Checkpoint SHA256: `a5469e7d1a2459075974f94420e554bf023c49d16d135912787ca6a4a64bc173`. See [`submission.json`](../submission.json) for the immutable Hugging Face revision.
