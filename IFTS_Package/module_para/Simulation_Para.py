import numpy as np
import torch
from os.path import join
from copy import deepcopy
from decimal import Decimal         
from IFTS_Package.base.Base_Para import Base_Para


class _Simu_Para(Base_Para):

    def __init__(self, rand_seed, config_path, *args, **kwargs):
        super().__init__()



