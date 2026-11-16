"""Official adapter for the frozen mixer_operating candidate."""
from pathlib import Path
from .submission_runtime import load_model, evaluate as eval, reset_model, encode_obs
from .model import ParaPred

def get_model(usr_args):
    return load_model(usr_args, ParaPred, 'mixer_operating', Path(__file__).resolve().parent)
