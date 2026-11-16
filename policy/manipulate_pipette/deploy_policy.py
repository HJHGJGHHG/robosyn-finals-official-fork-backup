"""Official adapter for the frozen manipulate_pipette candidate."""
from pathlib import Path
from .submission_runtime import load_model, evaluate as eval, reset_model, encode_obs
from .model import ParaPred

def get_model(usr_args):
    return load_model(usr_args, ParaPred, 'manipulate_pipette', Path(__file__).resolve().parent)
