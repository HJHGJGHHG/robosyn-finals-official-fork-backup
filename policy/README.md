# Initial RGB Perception and Template-Based Planning

RoboSynChallenge finals submission for all ten official tasks. Three initial RGB views predict task geometry; task templates and robot kinematics convert these predictions into joint commands.

**Authors:** Jiale Huang*, Zhenjie Liu*, MD AL ALIF, Xuandong Huang, Bingzhao Cai, and Shangfei Wang. University of Science and Technology of China. *Equal contribution.

**Code:** https://github.com/HJHGJGHHG/robosyn-finals

The repository preserves the official RoboSynChallenge revision `45d9b9c4c78c66ef340a055697f3ad833ce567f4`. All submission additions are under `policy/`. Each task directory contains `eval.sh`, `deploy_policy.yml`, `deploy_policy.py`, its model definition, planner, and static assets. The ten Hugging Face repositories listed below are reserved publication targets. **Checkpoint upload is pending a Hugging Face token with write permission.** The local files and checksums are prepared; revision fields remain empty until publication. Run the download commands after those revisions are populated.

## Install and download

```bash
git clone https://github.com/HJHGJGHHG/robosyn-finals.git
cd robosyn-finals
```

Install the official simulator, assets, and EmbodiChain dependencies using the [RoboSynChallenge installation guide](https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html). Then, in that Python environment:

```bash
python -m pip install -r policy/click_bell/requirements.txt
python -m pip install 'huggingface_hub>=0.27,<2'
python policy/download_checkpoints.py
python policy/download_checkpoints.py --verify-only
```

All ten tasks use the same dependency list. The tested loading environment uses Python 3.11.15, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, NumPy 2.2.6, and PyYAML 6.0.3. Downloading all checkpoints requires approximately 2.08 GB. Downloads are public and require no access token. The downloader pins each HF commit and checks SHA256 before installing it at `policy/<task>/checkpoint/best.pt`.

To download only one task:

```bash
python policy/download_checkpoints.py --task items_handover
```

## Submission fields

The following table can be copied into the submission form. [submission.csv](submission.csv) contains the same fields, and [submission.json](submission.json) additionally contains complete deployment configurations, checkpoint sizes, checksums, and immutable HF revisions.

Run commands are executed from the repository root. `0` is the example **physical GPU index**; replace it with the assigned GPU. The launcher sets both `gpu_id` and `pytorch_device` consistently. `random` selects the random condition; replace it with `clear` for clear evaluation. All task-specific template parameters are already in the listed YAML, so **no additional configuration overrides are required**.

| Task | Policy directory | HF checkpoint | Task run command | Configuration overrides |
|---|---|---|---|---|
| `click_bell` | `policy/click_bell` | [HJHGJGHHG/robosyn-click_bell](https://huggingface.co/HJHGJGHHG/robosyn-click_bell) | `bash policy/click_bell/eval.sh 0 random` | `setting=random`, `gpu_id=0`, `pytorch_device=cuda:0`; set by launcher |
| `drawer_open_place` | `policy/drawer_open_place` | [HJHGJGHHG/robosyn-drawer_open_place](https://huggingface.co/HJHGJGHHG/robosyn-drawer_open_place/tree/5f29e8443fa29b9d5797c6e9c8e4d185d95f625f) | `bash policy/drawer_open_place/eval.sh 0 random` | `setting=random`, `gpu_id=0`, `pytorch_device=cuda:0`; set by launcher |
| `handle_basket` | `policy/handle_basket` | [HJHGJGHHG/robosyn-handle_basket](https://huggingface.co/HJHGJGHHG/robosyn-handle_basket) | `bash policy/handle_basket/eval.sh 0 random` | `setting=random`, `gpu_id=0`, `pytorch_device=cuda:0`; set by launcher |
| `item_assembly` | `policy/item_assembly` | [HJHGJGHHG/robosyn-item_assembly](https://huggingface.co/HJHGJGHHG/robosyn-item_assembly) | `bash policy/item_assembly/eval.sh 0 random` | `setting=random`, `gpu_id=0`, `pytorch_device=cuda:0`; set by launcher |
| `items_handover` | `policy/items_handover` | [HJHGJGHHG/robosyn-items_handover](https://huggingface.co/HJHGJGHHG/robosyn-items_handover) | `bash policy/items_handover/eval.sh 0 random` | `setting=random`, `gpu_id=0`, `pytorch_device=cuda:0`; set by launcher |
| `manipulate_pipette` | `policy/manipulate_pipette` | [HJHGJGHHG/robosyn-manipulate_pipette](https://huggingface.co/HJHGJGHHG/robosyn-manipulate_pipette) | `bash policy/manipulate_pipette/eval.sh 0 random` | `setting=random`, `gpu_id=0`, `pytorch_device=cuda:0`; set by launcher |
| `mixer_operating` | `policy/mixer_operating` | [HJHGJGHHG/robosyn-mixer_operating](https://huggingface.co/HJHGJGHHG/robosyn-mixer_operating) | `bash policy/mixer_operating/eval.sh 0 random` | `setting=random`, `gpu_id=0`, `pytorch_device=cuda:0`; set by launcher |
| `sample_loading` | `policy/sample_loading` | [HJHGJGHHG/robosyn-sample_loading](https://huggingface.co/HJHGJGHHG/robosyn-sample_loading) | `bash policy/sample_loading/eval.sh 0 random` | `setting=random`, `gpu_id=0`, `pytorch_device=cuda:0`; set by launcher |
| `table_rearrangement` | `policy/table_rearrangement` | [HJHGJGHHG/robosyn-table_rearrangement](https://huggingface.co/HJHGJGHHG/robosyn-table_rearrangement) | `bash policy/table_rearrangement/eval.sh 0 random` | `setting=random`, `gpu_id=0`, `pytorch_device=cuda:0`; set by launcher |
| `water_pouring` | `policy/water_pouring` | [HJHGJGHHG/robosyn-water_pouring](https://huggingface.co/HJHGJGHHG/robosyn-water_pouring) | `bash policy/water_pouring/eval.sh 0 random` | `setting=random`, `gpu_id=0`, `pytorch_device=cuda:0`; set by launcher |

For every task, the HF checkpoint is `best.pt` at the repository root (subdirectory `.`). The policy configuration is `policy/<task>/deploy_policy.yml`; `checkpoint_path: checkpoint/best.pt` is resolved relative to that task directory. Published entries link to their pinned revision; pending entries list their planned repository URL.

## Arguments and configuration

```text
bash policy/<task>/eval.sh GPU_ID [random|clear] [--configuration_key value ...]
```

- `GPU_ID`: required physical GPU index; CUDA and Vulkan use the same numbering.
- `random|clear`: optional condition, defaults to `random`.
- `PYTHON_BIN`: optional environment variable selecting the Python interpreter.
- Extra arguments override YAML values, e.g. `--max_episodes 1` for a one-episode run.

```bash
bash policy/items_handover/eval.sh 0 clear --max_episodes 1
```

Shared YAML defaults are `seed: 161803`, `max_episodes: 20`, `max_steps: 1000`, `headless: true`, `eval_expert_check: false`, and `eval_reset_sync_steps: 100`. Video logging records the three cameras at 25 fps. The task launchers use the official evaluation loop with fixed table height and the configured material reset baseline. Expert screening is disabled.

Drawer uses the P6 template with `pull_distance_m: 0.185`, `timing: compact`, and `close_enabled: true`. Sample Loading uses `template: nominal_v1`, `control_variant: C0`, `transport_steps: 40`, `settle_steps: 25`, and `right_repair_enabled: true`. These values are provided by their YAML files and need no command-line overrides. Remaining task planning options are also supplied by each YAML.

## Policy interface

Each task provides `get_model(usr_args)`, `eval(env, model, observation)`, and `reset_model(model)`. Inputs are RGB from `cam_high`, `cam_right_wrist`, and `cam_left_wrist`, together with `observation['robot']['qpos']`. Actions contain left-arm six joints, left gripper, right-arm six joints, and right gripper. Arm targets are in radians; gripper actions are in meters (0–0.05), while observed gripper positions are normalized to 0–1.

The initial predictions parameterize the task template. Drawer additionally uses public joint feedback for its grasp and placement adjustments. `eval` sends one action per call and returns `(observation, info, truncated, inference_times)`; `reset_model` clears episode state. The official evaluator can load these adapters directly.

## Pinned checkpoints

| Task | HF revision | SHA256 |
|---|---|---|
| `click_bell` | `pending publication` | `b40d315ce09f4f95e9fb524137c6c5d0b891dafbfa9f32d7e94c2ea568c9ea83` |
| `drawer_open_place` | `5f29e8443fa29b9d5797c6e9c8e4d185d95f625f` | `cbc8b0fe6c523eb8a4091d761c4f2525ef90367581da7b2ea7ecd039943b00b2` |
| `handle_basket` | `pending publication` | `46b85eb5c84eea1e85eb05bd6f658d178b7ca3ce34247cad894015e4336640d2` |
| `item_assembly` | `pending publication` | `0063eff83785a462639a313b52a3b27c858b48258867c32bc59eba172880e22e` |
| `items_handover` | `pending publication` | `081dd60fea973d4d4fdcea4e84ba9714e1dd3b207037f30ead395c42cc184cb8` |
| `manipulate_pipette` | `pending publication` | `035f67916b37acb691c76a9c28fa811fff0e88c245d8801b1fc940cbc2a8ce41` |
| `mixer_operating` | `pending publication` | `d4c4da04d533066afa915c63e65e2f5d7c38b172b40aa42bedef1d17a4aef328` |
| `sample_loading` | `pending publication` | `fa2240b33ac7517021db8ce97972b8d339cd64e4f403da71637373bbe0398814` |
| `table_rearrangement` | `pending publication` | `5648df69c6ce03f08928c0c49ea303433483ebd192b9e37b5e6a5e5ea097f982` |
| `water_pouring` | `pending publication` | `a5469e7d1a2459075974f94420e554bf023c49d16d135912787ca6a4a64bc173` |
