import multiprocess as mp
import torch
from loguru import logger


def set_mp_start_method(method=None):
    if torch.cuda.is_available():
        desired_method = 'spawn'
    else:
        desired_method = method or mp.get_start_method(
            allow_none=True) or 'fork'

    try:
        mp.set_start_method(desired_method, force=True)
        logger.info(
            f"Setting multiprocess start method to '{desired_method}'.")
    except RuntimeError as e:
        logger.error(f'Error setting multiprocess start method: {e}')


def initialize(**kw_args):
    mp_start_method = kw_args.pop('mp_start_method', None)
    set_mp_start_method(mp_start_method)
