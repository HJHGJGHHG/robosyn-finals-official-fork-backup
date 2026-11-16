# item_assembly

Initial RGB perception and template-based planning for RoboSynChallenge.

- Hugging Face checkpoint: [HJHGJGHHG/robosyn-item_assembly](https://huggingface.co/HJHGJGHHG/robosyn-item_assembly).
- Policy directory: `policy/item_assembly`.
- Deployment configuration: [`deploy_policy.yml`](deploy_policy.yml).
- Required entrypoints: `get_model(usr_args)`, `eval(env, model, observation)`, `reset_model(model)` in `deploy_policy.py`.

From the repository root, after installing the [dependencies](../README.md):

```bash
python policy/download_checkpoints.py --task item_assembly
bash policy/item_assembly/eval.sh 0 random
```

Use a physical GPU index in place of `0`; `random` can be replaced with `clear`. Additional YAML overrides follow the setting, e.g. `--max_episodes 1`. No task-specific CLI overrides are required: all template parameters are already in `deploy_policy.yml`. The checkpoint path is relative to this policy directory: `checkpoint/best.pt`.

RGB inputs are `cam_high`, `cam_right_wrist`, and `cam_left_wrist`, together with public robot joint observations. The policy sends 14-dimensional absolute joint/gripper targets through the official environment interface. The task directory includes its model definition, planner, and static assets.

Checkpoint SHA256: `0063eff83785a462639a313b52a3b27c858b48258867c32bc59eba172880e22e`. See [`submission.json`](../submission.json) for the immutable Hugging Face revision.
