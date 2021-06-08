
import matplotlib.pyplot as plt
from torch.optim import optimizer
import libs.utils.globals as g
import os
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from libs.utils.utils import *
from libs.utils.logger import Logger
import time
import numpy as np
import warnings
import pdb

from libs.core.core_base import model as base_model

#-----------------------------------------------------

# This is there so that we can use source_import from the utils to import model
def get_core(*args):
    return base_model(*args)