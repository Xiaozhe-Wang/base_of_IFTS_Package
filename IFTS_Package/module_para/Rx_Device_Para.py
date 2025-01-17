import time
import numpy as np
from os.path import join
from copy import deepcopy


# import sys
# import os 
# sys.path
# sys.path.append(os.getcwd())

from IFTS_Package.module_para.Simulation_Para import _Simu_Para

from IFTS_Package.library.trans import ch,rx_device,rx_dsp,sig_rx,sig_tx,tx_device,tx_dsp


class _Rx_Device_Para(_Simu_Para):
    '''
        config_path是存放yml文件的目录，如：
        '/home/Chenmingzhe/cmz_simulation/IFTS_Package/user_info/Chenmingzhe/simulation/AWGN/trans/ch/'
    '''
    def __init__(self, rand_seed, config_path):
        super().__init__(rand_seed,config_path)
        self.set_module(path=config_path)

    '''
    李博、史博、肖博：
    补充一个__name__函数，name的名字与library下的.py文件名字相同
        例如：
            def __awgn__ (self,config):
            value=self.init_obj(config=config,module=ch.additive_noise)
            setattr(self, config['yml_name'], value)
            print(f"实例化后的awgn_np的SNR为{self.channel.SNR}")
    '''
    


# if __name__ == '__main__':
#     config_path='/home/Chenmingzhe/cmz_simulation/IFTS_Package/user_info/Chenmingzhe/simulation/AWGN/trans/ch/'
#     cmz=Sig_Tx_Para(rand_seed=10000,config_path=config_path)






