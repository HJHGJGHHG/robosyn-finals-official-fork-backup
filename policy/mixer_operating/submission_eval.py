"""Finals configuration adapter around the unmodified official evaluation loop.

Uses fixed table height and the configured material baseline. No expert precheck.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys


def finals_gym_config(config):
    """Copy a gym config, set 1000 steps and remove only table-height events."""
    result = deepcopy(config)
    result['max_episode_steps'] = 1000
    events = result.get('env', {}).get('events', {})
    for key in list(events):
        if 'table_height' in key:
            del events[key]
    return result


def capture_configured_reset_materials(env, gym_config):
    """Capture declared rigid-object materials as defaults before the first reset.

    Runs once after scene creation. Reset events still run after restoration,
    so reset/interval randomization is preserved. No physical state is changed.
    """
    from embodichain.lab.sim import VisualMaterialCfg
    base = env.unwrapped
    configs = {c['uid']: c['shape']['visual_material']
               for c in gym_config.get('rigid_object', []) + gym_config.get('background', [])
               if 'visual_material' in c.get('shape', {})}
    for event in gym_config.get('env', {}).get('events', {}).values():
        if event['func'] == 'set_rigid_object_visual_material' and event['mode'] == 'startup':
            configs[event['params']['entity_cfg']['uid']] = event['params']['mat_cfg']
    for uid, config in configs.items():
        obj = base.sim.get_rigid_object(uid)
        obj.set_visual_material(base.sim.create_visual_material(VisualMaterialCfg.from_dict(config)))
        # The SDK captured mesh materials before applying shape.visual_material.
        obj._initialize_existing_visual_material()
    base._submission_reset_material_uids = sorted(configs)
    return sorted(configs)


def load_evaluator(repo_root=None):
    """Load official helpers and install finals environment construction hooks."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    path = root / 'scripts/eval_policy.py'
    spec = importlib.util.spec_from_file_location('submission_official_evaluator', path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    original_find = evaluator.find_gym_config
    original_make = evaluator.make_env_from_configs
    original_recorder = evaluator.create_video_recorder

    def find_gym_config(config):
        if int(config.get('max_steps', 1000)) != 1000:
            raise ValueError('Finals evaluation requires max_steps=1000')
        return finals_gym_config(original_find(config))

    def make_env(config, gym_config, action_config):
        env, effective = original_make(config, gym_config, action_config)
        capture_configured_reset_materials(env, effective)
        return env, effective

    def create_video_recorder(config, run_dir):
        recorder = original_recorder(config, run_dir=run_dir)
        if recorder is not None and 'eval_video_fps' in config:
            fps = int(config['eval_video_fps'])
            if fps <= 0:
                raise ValueError('eval_video_fps must be positive')
            recorder.fps = fps
        return recorder

    evaluator.find_gym_config = find_gym_config
    evaluator.make_env_from_configs = make_env
    evaluator.create_video_recorder = create_video_recorder
    return evaluator



def main():
    evaluator = load_evaluator()
    original_parse = evaluator.parse_args_and_config
    def parse_config():
        config = original_parse()
        if config.get('eval_expert_check', False):
            raise ValueError('Submission inference does not use expert prechecks')
        config['eval_expert_check'] = False
        return config
    evaluator.parse_args_and_config = parse_config
    evaluator.main()


if __name__ == '__main__':
    main()
