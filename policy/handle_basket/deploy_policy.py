"""Official adapter for the frozen handle_basket candidate."""
from pathlib import Path
from .submission_runtime import load_model, evaluate as eval, reset_model, encode_obs
from .model import ParaPred

def get_model(usr_args):
    return load_model(usr_args, ParaPred, 'handle_basket', Path(__file__).resolve().parent)
