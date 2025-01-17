import os
import numpy as np
from sys import path

__all__ = [
    "library",
    # "base",
    # "core_lib",
    # "function",
    # "library",
    # "tools",
    # "module_main",
    # "module_para",
    # "visualization",
]
# __version__ = "1.0.0"   # 0.0.1 old version, 1.0.0: pro version

# def get_version():
#     return __version__

print('======== Intelligent Fiber Transmission System (IFTS) ========')
code_dir = os.getcwd()                      #返回当前工作目录
path.append(code_dir)
path.append(os.path.join(code_dir, 'IFTS_Package'))
# print(f' Code Dir: {code_dir}')
# print(f' Version: {get_version()}')
# print(' Add path ...')
# print(' Import config_dir ...')
# from IFTS_Package.tools.config_dir import mkdir, ch_modeling_test_path_init, ch_modeling_dataset_path_init
# from IFTS_Package.tools.path.config_dir import path_log_init
# print(' Import tools ...')
# from IFTS_Package.tools.utilis import version_select, init_ftn, init_obj, setup_seed, setup_config, time_record
# from IFTS_Package.tools.communication import calculation
print(' Import module_para ...')
from IFTS_Package.module_para import Ch_Para,Rx_Device_Para,Rx_Dsp_Para,Sig_Rx_Para,Sig_Tx_Para,Simulation_Para,Tx_Device_Para,Tx_Dsp_Para
# print(' Import module_main ...')
# from IFTS_Package.module_main.trans import tx_main, sig_main, txdsp_main, rxdsp_main, channel_main, rxdsp_parallel, DRNN_main, rxdsp_ofdm

print('====== Copyright by SJTU LIFE Lab. All rights reseverd. ======')
