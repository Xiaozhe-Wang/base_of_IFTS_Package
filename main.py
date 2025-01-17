import IFTS_Package as IFTS

rand_seed=10000
config_path = '/home/Chenmingzhe/cmz_simulation/IFTS_Package/user_info/Chenmingzhe/simulation/trans/' 
cmz=IFTS.Sig_Tx_Para._Sig_Tx_Para(rand_seed,config_path)
print(f"yml文件实例化：{cmz.channel} {cmz.channel_2}")
print(f"yml文件实例化属性SNR：{cmz.channel.SNR}")
print(f"yml_2文件实例化属性SNR：{cmz.channel_2.SNR}")

config_path='/home/Chenmingzhe/cmz_simulation/IFTS_Package/user_info/Chenmingzhe/simulation/trans/ch/additive_noise/channel.yml'
cmz=IFTS.Sig_Tx_Para._Sig_Tx_Para(rand_seed,config_path)
print(f"yml文件实例化：{cmz.channel}")

config_path='/home/Chenmingzhe/cmz_simulation/IFTS_Package/user_info/Chenmingzhe/simulation/trans/ch/channel_2.yml'
cmz=IFTS.Sig_Tx_Para._Sig_Tx_Para(rand_seed,config_path)
print(f"yml_2文件实例化：{cmz.channel_2}")







