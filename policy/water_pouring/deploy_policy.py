"""Official adapter for the verified water_pouring policy."""
from pathlib import Path
from .submission_runtime import load_model, evaluate as eval, reset_model, encode_obs
from .model import ParaPred


def get_model(usr_args):
    return load_model(usr_args, ParaPred, "water_pouring", Path(__file__).resolve().parent)
