import time
import numpy as np
from os.path import join
from copy import deepcopy


import sys
import os 
sys.path
sys.path.append(os.getcwd())

from IFTS_Package.module_para.Simulation_Para import _Simu_Para

from IFTS_Package.library.trans import ch,rx_device,rx_dsp,sig_rx,sig_tx,tx_device,tx_dsp

class _Sig_Tx_Para(_Simu_Para):
    
    def __init__(self, rand_seed, config_path):
        super().__init__(rand_seed,config_path)
        self.set_module(path=config_path)

    def __awgn__ (self,config):
        value=self.init_obj(config=config,module=ch.additive_noise)
        setattr(self, config['yml_name'], value)

    def __awgn_2__ (self,config):
        value=self.init_obj(config=config,module=ch)
        setattr(self, config['yml_name'], value)






