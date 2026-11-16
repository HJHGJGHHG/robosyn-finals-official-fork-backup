"""Official adapter for the frozen table_rearrangement candidate."""
from pathlib import Path
from .submission_runtime import load_model, evaluate as eval, reset_model, encode_obs
from .model import ParaPred

def get_model(usr_args):
    return load_model(usr_args, ParaPred, 'table_rearrangement', Path(__file__).resolve().parent)
