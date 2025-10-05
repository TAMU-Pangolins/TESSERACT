import numpy as np
from .constants import EV_TO_J

def ev_to_j(x):
    return np.asarray(x, dtype=float) * EV_TO_J
