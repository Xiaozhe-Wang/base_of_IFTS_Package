import time
import numpy as np
import torch
from torch.fft import fft, ifft, fftshift, ifftshift
from copy import deepcopy
from IFTS_Package.module_para.signal_para import Sig_Para
from IFTS_Package.core_lib.fiber_simulation import rx_dsp as rx 
from IFTS_Package.core_lib.fiber_simulation import calcu

class Rx_para(Sig_Para):
    def __init__(self, rand_seed, simu_configs):
        r"""
            Check receiver's config parameters.
            Initialize functional modules.
            Save config results.
        """
        super().__init__(rand_seed, deepcopy(simu_configs))
        configs = deepcopy(simu_configs['Rx_Para'])
        start_t1 = time.time() 
        self.dsp_mode    = self._check('dsp_mode')[1]
        self.infor_print    = self._check('infor_print_arr', np.ones(3))[3]
        self.fig_plot       = self._check('fig_plot_arr', np.ones(3))[3]
        self.save_data      = self._check('save_data_arr', np.zeros(3))[3]
        self.print_buff     = []
        self.print_buff.append('info__ rx_para initializing ...')
        
        if self._check('data_mode', 'hybrid') == 'hybrid':
            self.rx_data_mode = 'torch'
        else:
            self.rx_data_mode = self.data_mode         # numpy, torch
        self._check('fiber_config', configs=simu_configs['Ch_Para'])
        # self.oscop_sam_num  = self._check('oscop_sam_num', 2000000, configs)
        self.frame_num  = self._check('frame_num', 1, configs)
        self.sam_rate   = self.oscope_sam_rate = self._check("rx_sam_rate")    # GHz
        if self._check('scm'):
            self.sym_rate   = self._check("sc_rate")    # GHz   ns = 1/GHz
        else:
            self.sym_rate   = self._check("sym_rate")   # GHz   ns = 1/GHz
        self.dt         = 1000 / self.sam_rate          # ps ps = 1000 ns
        self.sym_time   = 1000 / self.sym_rate
        self.awg_memory_depth = self._check("awg_memory_depth")
        self.frame_sym_num = int(np.ceil(self.awg_memory_depth / (self.tx_sam_rate / self.sym_rate)))
        # self.frame_sym_num = self.sym_num_per_pol
        self.sym_simu_time = np.arange(0, self.frame_sym_num) * self.sym_time
        self.sam_now = self.upsam = (self.sam_rate)  / self.sym_rate
        #####
        #self.rx_block_upsam = 2
        #self.rx_dsam_rate = self.upsam/self.rx_block_upsam   
        #####
        r"""
        Coarse LPF
            |
        IQ Imbalance
            |            
        Chromatic Dispersion
            |
        MIMO (or Muti-sample Nonlinearity Compensation)
            |
        Fine LOFO
            |
        Synchronization
            |
        Carrier Phase Estimation
        """
        if self._check('ckl_com', 1, configs):
            self._check('ckl_com_config', configs = configs)
            self._check_data_mode(self.ckl_com_config, data_mode='numpy')
            self.__clock_leakage__()
        
        if self._check('iq_deskew', 1, configs):
            self._check('iq_deskew_config', configs = configs)
            self._check_data_mode(self.iq_deskew_config, data_mode='torch')
            self.__iq_deskew__()
            
        if self._check('lpf', 1, configs):
            # brickwall, bessel, butter, gaussian, rc, rrc
            self._check('lpf_config', configs = configs) 
            self._check_data_mode(self.lpf_config, data_mode='torch')
            self.__low_pass_filter__()
        
        if self._check('iq_balance', 1, configs):
            self._check('iq_balance_config', configs = configs)
            self._check_data_mode(self.iq_balance_config, data_mode='numpy')
            self.__iq_balance__()
        
        if self.load_len:
            self._check('len_config_path', configs = self.fiber_config)
            self.len_config = self.read_configs(self.len_config_path)
            self.len_array = np.zeros((self.span_num))
            for i in range(self.span_num):
                self.len_array[i] = self.len_config['span' + str(i+1)]
            self.total_len = np.sum(self.len_array)
        else:
            # Span length calculation, also can set different length in each span
            self.span_num = int(self.total_len / self.span_len)
            if self.span_num * self.span_len != self.total_len:
                self.span_num = self.span_num + 1
                self.len_array = np.zeros(self.span_num) + self.span_len
                self.len_array[self.span_num-1] = self.total_len - (self.span_num - 1) * self.span_len
            else:
                self.len_array = np.zeros(self.span_num) + self.span_len
            if np.sum(self.len_array) != self.total_len:
                raise ValueError('The sum of span length is different with total length !')

        if self._check('nl_com', 1, configs):
            self._check('nl_com_config', configs = configs)
            self._check_data_mode(self.nl_com_config, data_mode='torch')
            self.__get_channel_para__()
            self.__nolinearity_compensation__()
        
        if self._check('cdcom', 1, configs):
            if simu_configs['Simu_Para']['pre_cdcom']:
                pre_cd_len = simu_configs['Tx_Para']['pre_cdcom_config']['cd_len']
            else:
                pre_cd_len = 0
            self._check('cd_com_config', configs = configs) 
            self._check_data_mode(self.cd_com_config, data_mode='torch')
            self.__cdcom__(pre_cd_len)

        if self._check('cdpmdcom', 0, configs):
            self._check('cd_com_config', configs = configs) 
            self._check_data_mode(self.cd_com_config, data_mode='torch')
            self.__cdpmdcom__()

        if self._check('mimo', 1, configs):
            self._check('mimo_config',  configs = configs)   
            self._check_data_mode(self.mimo_config, data_mode='numpy')
            self.__mimo__()
            
        if self._check('clofo', 1, configs):
            self._check('clofo_config', configs = configs)   
            self._check_data_mode(self.clofo_config, data_mode='torch')
            self.__clofo__()

        if self._check('synchronization', 1, configs):
            self._check('synchron_config', configs = configs)   # 2x2, 4x4
            self._check_data_mode(self.synchron_config, data_mode='numpy')
            self.__synchronization__()

        if self._check('cpe', 1, configs):
            self._check('cpe_config', configs = configs)   # bps, coarse
            self._check_data_mode(self.cpe_config, data_mode='numpy')
            self.__cpe__()

        if self._check('real_mimo', 1, configs):
            self._check('real_mimo_config',  configs = configs)   
            self._check_data_mode(self.real_mimo_config, data_mode='numpy')
            self.__real_mimo__()
        
        if self._check('pertubation', 0, configs):
            self._check('pertubation_config',  configs = configs)   
            self._check_data_mode(self.pertubation_config, data_mode='numpy')
            self.__get_channel_para__()
            self.__pertubation__()
        
        self.rx_config = configs
        start_t2 = time.time() 
        self.print_buff.append('info__ rx_para initialized with {:.2f} s'.format(start_t2-start_t1))
        
        # self.save_configs(self.result_path + 'Rx_Para.yaml', configs)

    def __clock_leakage__(self):
        config = self.ckl_com_config
        self._config_check('leakage_freq', [-25, -12.5, 0, 12.5, 25], config)
        self._config_check('leakage_comp', np.zeros((4, 5, 2)), config)
        self._config_check('sam_rate', self.sam_rate, config)   
        self._config_check('fft_num', self.oscop_sam_num, config)   
        self._config_check('data_mode', 'numpy', config)   
        self._config_check('device', self.device, config)
        self.ckl_com_obj = rx.cl_com_design(**config)

    def __iq_balance__(self):
        mode = self.iq_balance_config['type']
        self.iq_balance_obj = rx.imbalance_eq.IQ_Balance(mode, data_mode = 'torch', device = self.device)

    def __iq_deskew__(self):
        config = self.iq_deskew_config
        self._config_check('mode', 'freq', config)   
        self._config_check('data_mode', 'numpy', config)   
        self._config_check('device', self.device, config)   
        self.iq_deskew_obj = rx.skew_eq.Skew_Equalization(**config)
        if config['mode'] == 'freq':
            self._config_check('sam_rate', self.sam_rate, config['args'])
        self._config_check('skew_list', [0, 0, 0, 0], config['args'])
        self.iq_deskew_obj.init(**config['args'])

    def __low_pass_filter__(self):
        r"""
            LPF initialization function
            optional LPF type: brickwall, bessel, butter, gaussian, rc, rrc
        """
        # brickwall, bessel, butter, gaussian, rc, rrc
        config = self.lpf_config
        self._config_check('mode', 'freq', config)   
        self._config_check('sam_rate', self.sam_rate, config)   
        self._config_check('fft_num', self.oscop_sam_num, config)   
        self._config_check('data_mode', 'torch', config)   
        self._config_check('device', self.device, config)  
        self.lpf_obj = rx.lpf_design.LPF(**config)

        lpf_args = config['args']
        max_lofo = self._config_check('max_lofo', 0.5, lpf_args)
        roll_off = self._config_check('roll_off', 0.1, lpf_args)
        filter_bw = self._config_check('filter_bw',\
            self.sym_rate * (1 + roll_off) / 2 + max_lofo,lpf_args)
        if self.lpf_obj.mode == 'rrc':
            self._config_check('beta', roll_off, lpf_args)
            self._config_check('upsam', self.sam_now, lpf_args)
            if max_lofo != 0:
                self.print_buff.append('warning__    rx_para LPF:RRC will filter the signal becasue of LOFO at rx_para')
                # self.logger.warn('RRC will filter the signal becasue of LOFO at rx_para')
        else:
            if filter_bw > self.channel_space:
                filter_bw = self.channel_space
            if self.lpf_obj.mode == 'bessel':
                self._config_check('order', 15, lpf_args)     # bessel para
            elif self.lpf_obj.mode == 'butter':
                self._config_check('gpass', 5, lpf_args)     # butterworth para
                self._config_check('gstop', 40, lpf_args)    # butterworth para
            self._config_check('cut_off', filter_bw, lpf_args)
            if self.infor_print:
                # self.logger.info("    rx_para LPF: {} filter and bandwidth {:.2f} GHz".format(config['mode'],filter_bw))
                self.print_buff.append('info__    rx_para LPF: {} filter and bandwidth {:.2f} GHz'.format(config['mode'],filter_bw))
        self._config_check('comp_s21', 1, lpf_args)
        self._config_check('s21_bw', filter_bw, lpf_args)
        self._config_check('s21_path', configs=lpf_args)
        self.lpf_obj.init(**lpf_args)

    def __get_channel_para__(self):
        wavelength = self._check('wavelength', 1550, self.fiber_config)
        # Calculate dispersion parameters
        D = self._config_check('D', 17, self.fiber_config)    # 16.75  dispersion Parameter (ps/nm.km); if anomalous dispersion(for compensation),D is negative
        S = self._config_check('S', 0.075, self.fiber_config)    # S slope ps/(nm^2.km)
        beta2c = self._config_check('beta2c',\
            - 1000 * D * wavelength ** 2 / (2 * self.Constant_pi * self.Constant_C), self.fiber_config) # beta2 (ps^2/km);
        beta3c = self._config_check('beta3c', 0, self.fiber_config)
        # self.beta3c = 10 ** 6 * (self.S - self.beta2c * (4 *self.Constant_pi * C / self.wavelength ** 3) / 1000) / (2 *self.Constant_pi * C / self.wavelength ** 2) ** 2 # (ps^3/km)
        self.df = ((np.arange(1, self.channel_num + 1) - (self.channel_num + 1) / 2) * self.channel_space)
        self.beta0 = beta2c / 2 * (2 * self.Constant_pi * self.df)**2 * 10**-6 + beta3c / 6 * (2 * self.Constant_pi * self.df)**3 * 10**-9   #WDM的接收端的cdc与df有关，所以要更新参数
        self.beta1 = beta2c * 2 * self.Constant_pi * self.df * 10**-3 + beta3c * (2 * self.Constant_pi * self.df)**2 * 10**-6 / 2      
        self.beta2 = beta2c + 2 * self.Constant_pi * self.df * beta3c * 10**-3
        self.beta3 = beta3c
        n2 = self._config_check('n2', 2.6e-20, self.fiber_config)
        Aeff = self._config_check('Aeff', 80, self.fiber_config)
        self.gamma = 1000 * 2 *self.Constant_pi * n2 / (wavelength * 10 ** -9 * Aeff * 10**-12)
        self.sam_per_sym = int(self.ch_sam_rate/(self.channel_num*self.sym_rate))
        self.alpha_indB = self._check('alpha_indB', 0.2,configs = self.fiber_config)
        self.alpha_loss = np.log(10 ** (self.alpha_indB / 10))

    def __nolinearity_compensation__(self):   
        self.rx_block_upsam =  2 
        self.rx_sam = int(self.frame_num * self.frame_sym_num * self.rx_block_upsam)
        self.rx_dsam_rate = self.upsam/self.rx_block_upsam   
        if self.nl_com_config['type'] == 'DBP':
            self.dbp = 1
            self.cdcom = 0
            self.dbp_sam = self._config_check('sym_sam', 2, self.nl_com_config['args'])  
            self.block_upsam = self.dbp_sam
            self.fft_num = self.sym_num_per_pol * self.dbp_sam
            self.pmd = 0
            self.sps = self._config_check('sps', 50, self.nl_com_config['args'])  
            self.nl_coeff = self._config_check('nl_coeff',1, self.nl_com_config['args'])   # decay coeff
            self.gamma = self.gamma * self.nl_coeff
            self.dz_con     = self.span_len / self.sps     # distance stepsize (km)
            self.dz_mode    = self._config_check('dz_mode', 'c', self.nl_com_config['args']) 
            self.edfa_gain = -self.alpha_indB * self.span_len                  # np: nonlinear phase
            
            self.__cal_digital_freq__()
            self.__step_size_para__()
            self.dbp_obj = rx.dbp_design.DBP(self.gamma,self.alpha_loss,self.edfa_gain,self.pmd,self.sig_power_dbm,\
                    self.phase_factor_freq,self.step_config,data_mode=self.rx_data_mode,device = self.device)
            
    def __cal_digital_freq__(self):
            
        '''
        beta(w+dwk)=b0 + b1k w + b2k w^2/2 + b3k w^3 /6 + ...
        b1k = d(beta(w))/dw = b2 dwk + b3 dwk^2 / 2
        b2k = d^2(beta(w))/dw^2
        '''
        sam_rate = self.sym_rate * self.dbp_sam
        self.w = 2 * self.Constant_pi * calcu.digital_freq_wshift(self.fft_num, sam_rate * 1e-3)
        self.phase_factor_freq = self.beta0[self.cut_idx]\
            + self.beta1[self.cut_idx] * (self.w)\
                + self.beta2[self.cut_idx] * (self.w ** 2) / 2\
                    + self.beta3 * (self.w ** 3) / 6
        if self.rx_data_mode == 'tensor':
            self.phase_factor_freq = torch.from_numpy(self.phase_factor_freq).to(self.device)

    def __cdcom__(self, pre_cd_len):
        r"""
            CDC initialization function
            Calculate and check dispersion parameters
        """
        wavelength = self._check('wavelength', 1550, self.fiber_config) # wavelength (nm)=10^-9(m)
        # Dispersion parameters
        D = self._config_check('D', 17, self.fiber_config) # 16.75  dispersion Parameter (ps/nm.km); if anomalous dispersion(for compensation),D is negative
        S = self._config_check('S', 0.075, self.fiber_config) # S slope ps/(nm^2.km)
        beta2c = self._config_check('beta2c',\
            - 1000 * D * wavelength ** 2 / (2 * self.Constant_pi * self.Constant_C), self.fiber_config) # beta2 (ps^2/km);
        beta3c = self._config_check('beta3c', 0, self.fiber_config)
        # self.beta3c = 10 ** 6 * (self.S - self.beta2c * (4 *self.Constant_pi * C / self.wavelength ** 3) / 1000) / (2 *self.Constant_pi * C / self.wavelength ** 2) ** 2 # (ps^3/km)
        if self.parallel_dsp:
            self.df_sc = ((np.arange(1, self.sc_num + 1) - (self.sc_num + 1) / 2) * self.sc_space)
            # self.df_sc = np.linspace(-(self.sc_num/2-1), self.sc_num/2, self.sc_num)* self.sc_space
            self.df_ch = ((np.arange(1, self.channel_num + 1) - (self.channel_num + 1) / 2) * self.channel_space).reshape(-1,1)
            self.df = self.df_ch + self.df_sc
        else:
            self.df = ((np.arange(1, self.channel_num + 1) - (self.channel_num + 1) / 2) * self.channel_space)
        # self.df = ((np.arange(1, self.channel_num + 1) - (self.channel_num + 1) / 2) * self.channel_space)
        self.beta0 = beta2c / 2 * (2 * self.Constant_pi * self.df)**2 * 10**-6 + beta3c / 6 * (2 * self.Constant_pi * self.df)**3 * 10**-9   #WDM的接收端的cdc与df有关，所以要更新参数
        self.beta1 = beta2c * 2 * self.Constant_pi * self.df * 10**-3 + beta3c * (2 * self.Constant_pi * self.df)**2 * 10**-6 / 2      
        self.beta2 = beta2c + 2 * self.Constant_pi * self.df * beta3c * 10**-3
        self.beta3 = beta3c

        config = self.cd_com_config
        self.rx_block_upsam = sam_per_sym = self._config_check('sam_per_sym', 2, config)
        self.rx_block_sam_rate = sam_rate = self.sym_rate * sam_per_sym
        self.rx_dsam_rate = self.upsam/self.rx_block_upsam   
        self.rx_sam = int(self.frame_num * self.frame_sym_num * self.rx_block_upsam)
        
        fft_num = self.frame_num * self.frame_sym_num * sam_per_sym
        beta = [self.beta0, self.beta1, self.beta2, self.beta3]
        # 
        self.cdcom_obj = rx.cdc_design.CDC(mode = config['type'],\
            beta = beta, sam_per_sym = sam_per_sym,\
                sam_rate = sam_rate, fft_num = fft_num,\
                    data_mode = 'torch', device = self.device)
        cdc_length = self._config_check('cdc_length', 'auto', config['args']) 
        self._config_check('cut_idx', self.cut_idx, config['args']) 
        
        if cdc_length == 'auto':
            cdc_length = self.total_len + self.pre_cd_len
            config['args']['cdc_length'] = cdc_length
        else:
            cdc_length = self._config_check('cdc_length', self.total_len, config['args'])
        self.cdc_length = cdc_length
        config['args']['parallel_dsp'] = self.parallel_dsp
        config['args']['sc_num'] = self.sc_num
        config['args']['nPol'] = self.nPol
        self.cdcom_obj.init(**config['args'])

    def __cdpmdcom__(self):
        r"""
            CDPMDC initialization function
            Calculate and check dispersion parameters
        """
        wavelength = self._check('wavelength', 1550, self.fiber_config) # wavelength (nm)=10^-9(m)
        # Dispersion parameters
        D = self._config_check('D', 17, self.fiber_config) # 16.75  dispersion Parameter (ps/nm.km); if anomalous dispersion(for compensation),D is negative
        S = self._config_check('S', 0.075, self.fiber_config) # S slope ps/(nm^2.km)
        beta2c = self._config_check('beta2c',\
            - 1000 * D * wavelength ** 2 / (2 * self.Constant_pi * self.Constant_C), self.fiber_config) # beta2 (ps^2/km);
        beta3c = self._config_check('beta3c', 0, self.fiber_config)
        # self.beta3c = 10 ** 6 * (self.S - self.beta2c * (4 *self.Constant_pi * C / self.wavelength ** 3) / 1000) / (2 *self.Constant_pi * C / self.wavelength ** 2) ** 2 # (ps^3/km)
        self.df = ((np.arange(1, self.channel_num + 1) - (self.channel_num + 1) / 2) * self.channel_space)
        self.beta0 = beta2c / 2 * (2 * self.Constant_pi * self.df)**2 * 10**-6 + beta3c / 6 * (2 * self.Constant_pi * self.df)**3 * 10**-9   #WDM的接收端的cdc与df有关，所以要更新参数
        self.beta1 = beta2c * 2 * self.Constant_pi * self.df * 10**-3 + beta3c * (2 * self.Constant_pi * self.df)**2 * 10**-6 / 2      
        self.beta2 = beta2c + 2 * self.Constant_pi * self.df * beta3c * 10**-3
        self.beta3 = beta3c

        pmd_config = self._check('pmd_config', self.fiber_config['args']['pmd_config'])
        config = self.cd_com_config
        
        self.rx_block_upsam = sam_per_sym = self._config_check('sam_per_sym', 2, config)
        self.rx_block_sam_rate = sam_rate = self.sym_rate * sam_per_sym
        self.rx_dsam_rate = self.upsam/self.rx_block_upsam   
        self.rx_sam = int(self.frame_num * self.frame_sym_num * self.rx_block_upsam)
        
        fft_num = self.frame_num * self.frame_sym_num * sam_per_sym
        beta = [self.beta0, self.beta1, self.beta2, self.beta3]
        self.cdpmd_com_obj = rx.cdpmdc_design.CDPMDC(\
            beta = beta, sam_per_sym = sam_per_sym,\
                sam_rate = sam_rate, fft_num = fft_num, pmd_config = pmd_config,\
                    data_mode = 'torch', device = self.device)
        
        cdc_length = self._config_check('cdc_length', 'auto', config['args']) 
        self._config_check('cut_idx', self.cut_idx, config['args']) 
        
        if cdc_length == 'auto':
            cdc_length = self.total_len + self.pre_cd_len
            config['args']['cdc_length'] = cdc_length
        else:
            cdc_length = self._config_check('cdc_length', self.total_len, config['args'])
        self.cdc_length = cdc_length
        self.cdpmd_com_obj.init(**config['args'])
        
    def __synchronization__(self):
        r"""
            Synchronization initialization function
            check synchronization's config parameters
        """
        config = self.synchron_config
        self._config_check('mode', '4x4', config)   
        self._config_check('frame_num', self.frame_num, config)   
        self._config_check('frame_size', self.frame_sym_num, config)   
        self._config_check('data_mode', 'torch', config)   
        self._config_check('device', self.device, config)  
        self.synchron_obj = rx.synchron_design.synchron(**config)

        synchron_args = self.synchron_config['args']
        self._config_check('parallel_dsp', self.parallel_dsp, synchron_args)
        self.synchron_obj.init(**synchron_args)
     
    def __mimo__(self):
        r"""
            MIMO initialization function
            check MIMO's config parameters
        """

        self._config_check('mode', 'TD_2x2', self.mimo_config)   
        self._config_check('lr_optim', 'constant', self.mimo_config)   
        self._config_check('half_taps_num', 15, self.mimo_config)   
        self._config_check('out_sym_num', 1, self.mimo_config)   
        self._config_check('upsam', 2, self.mimo_config)   
        self._config_check('block_size', 1, self.mimo_config)   
        self._config_check('data_mode', 'numpy', self.mimo_config)   
        self._config_check('device', self.device, self.mimo_config)   
        self._config_check('infor_print', self.infor_print, self.mimo_config)   
        self.mimo_obj = rx.adaptive_filter_design.Vanilla(**self.mimo_config)
        
        mimo_args = self.mimo_config['args']
        self._config_check('algo_type', 'cma', mimo_args)  # cma, mma, dd_lms
        self._config_check('cma_pretrain', 1, mimo_args)  
        self._config_check('pre_train_iter', 2, mimo_args)  
        self._config_check('train_num', self.rx_sam, mimo_args)  
        # self._config_check('train_num', sam_num, mimo_args)  
        self._config_check('train_epoch', 2, mimo_args)  
        self._config_check('lr', 5.0e-4, mimo_args)  
        self._config_check('sam_num', self.rx_sam, mimo_args) 
        self._config_check('tap_init_mode', 1, mimo_args) 
        self._config_check('tap_init_value', 1.0, mimo_args) 
        self._config_check('radius_idx', -1, mimo_args) 
        self._config_check('parallel_dsp', self.parallel_dsp, mimo_args)
        self._config_check('sc_num', self.sc_num, mimo_args)
        self._config_check('nPol', self.nPol, mimo_args)
        self.mimo_obj.init(**mimo_args)


    def __clofo__(self):
        self._config_check('mode', 'freq_method', self.clofo_config)   
        self._config_check('sym_rate', self.sym_rate, self.clofo_config)   
        self._config_check('window_size', 61, self.clofo_config)   
        self._config_check('block_num', 6, self.clofo_config)   
        self._config_check('parallelism', 16, self.clofo_config)   
        self._config_check('infor_print', self.infor_print, self.clofo_config)   
        self._config_check('data_mode', 'torch', self.clofo_config)   
        self._config_check('device', self.device, self.clofo_config)     
        self.clofo_obj = rx.foe_design.FOE(**self.clofo_config)

        clofo_args = self.clofo_config['args']
        self.clofo_obj.init(**clofo_args)
    
    def __cpe__(self):
        r"""
            CPE initialization function
            check CPE's config parameters
        """
        self._config_check('mode', 'bps', self.cpe_config)   
        self._config_check('sym_rate', self.sym_rate, self.cpe_config)   
        self._config_check('window_size', 61, self.cpe_config)   
        self._config_check('block_num', 6, self.cpe_config)   
        self._config_check('parallelism', 1, self.cpe_config)   
        self._config_check('infor_print', self.infor_print, self.cpe_config)   
        self._config_check('data_mode', 'torch', self.cpe_config)   
        self._config_check('device', self.device, self.cpe_config) 
        self._config_check('parallel_dsp', self.parallel_dsp, self.cpe_config)   
        self.cpe_obj = rx.cpe_design.CPE(**self.cpe_config)
        
        cpe_args = self.cpe_config['args']
        if self.cpe_config['mode'] == 'vv':
            self._config_check('ml', 1, cpe_args)  
        elif self.cpe_config['mode'] == 'bps':
            self._config_check('phi_int', np.pi/2, cpe_args)  
            self._config_check('ml', 1, cpe_args) 
            self._config_check('data_aided', 1, cpe_args)   
        self.cpe_obj.init(**cpe_args)

    def __real_mimo__(self):

        self._config_check('mode', 'TD_4x4', self.real_mimo_config)   
        self._config_check('lr_optim', 'constant', self.real_mimo_config)   
        self._config_check('half_taps_num', 15, self.real_mimo_config)   
        self._config_check('out_sym_num', 1, self.real_mimo_config)   
        self._config_check('upsam', 1, self.real_mimo_config)   
        self._config_check('block_size', 1, self.real_mimo_config)   
        self._config_check('data_mode', 'numpy', self.real_mimo_config)   
        self._config_check('device', self.device, self.real_mimo_config)   
        self._config_check('infor_print', self.infor_print, self.real_mimo_config)   
        self.real_mimo_obj = rx.adaptive_filter_design.Vanilla(**self.real_mimo_config)
        
        mimo_args = self.real_mimo_config['args']
        upsam = self._config_check('upsam', 1, mimo_args) 
        sam_num = self.frame_sym_num * upsam
        self._config_check('algo_type', 'lms', mimo_args)  # cma, mma, dd_lms
        self._config_check('cma_pretrain', 0, mimo_args)  
        self._config_check('pre_train_iter', 0, mimo_args)  
        self._config_check('train_num', 65536, mimo_args)  
        self._config_check('train_epoch', 2, mimo_args)  
        self._config_check('lr', 5.0e-4, mimo_args)  
        self._config_check('sam_num', sam_num, mimo_args) 
        self._config_check('tap_init_mode', 1, mimo_args) 
        self._config_check('tap_init_value', 1.0, mimo_args) 
        self.real_mimo_obj.init(**mimo_args)
    
    def __pertubation__(self):
        # pertubation_config = self.pertubation_config['args']
        self._config_check('gamma', self.gamma, self.pertubation_config)
        self._config_check('beta2c', self.beta2, self.pertubation_config)
        self._config_check('tau', self.sym_time, self.pertubation_config)
        self._config_check('span_length', self.total_len, self.pertubation_config)
        self._config_check('sym_rate', self.sym_rate, self.pertubation_config)
        self._config_check('power', self.sig_power_dbm, self.pertubation_config)
        self.pertubation_obj = rx.pertubation_design.Pertubation(self.pertubation_config)
        self.pertubation_obj.init()

    def __step_size_para__(self):
        """Config for step size calculation in DBP.
        
        The method creates a Step_Size object for step size calculation. 
        Arguments required are wrapped up as a configuration dict and passed in to
        intialize the object. Refer to Step_Size.__init__ for details.

        """
        self.step_config = {}
        self.step_config['nPol'] = self.nPol
        self.step_config['nl_gamma'] = self.gamma
        self.step_config['alpha_loss'] = self.alpha_loss
        self.step_config['span_len'] = self.span_len
        self.step_config['pmd'] = 0
        self.step_config['dz_mode'] = 'c'
        self.step_config['constant_step_size'] = self.dz_con
    


class LDSP_para(Sig_Para):
    def __init__(self, rand_seed, simu_configs):
        super().__init__(rand_seed, deepcopy(simu_configs))
        configs = deepcopy(simu_configs['Rx_Para'])
        start_t1 = time.time() 
        self.dsp_mode    = self._check('dsp_mode')[1]
        self.infor_print    = self._check('infor_print_arr', np.ones(3))[3]
        self.fig_plot       = self._check('fig_plot_arr', np.ones(3))[3]
        self.save_data      = self._check('save_data_arr', np.zeros(3))[3]
        self.print_buff     = []
        self.print_buff.append('info__ rx_para initializing ...')
        
        if self._check('data_mode', 'hybrid') == 'hybrid':
            self.rx_data_mode = 'torch'               
        else:
            self.rx_data_mode = self.data_mode         # numpy, torch
        self._check('fiber_config', configs=simu_configs['Ch_Para'])

        """
            Hype-paramters of learned DSP
                lr, block_size, block_num
        """
        # self.ldsp_obj = rx.ldsp_design.LDSP()
        self._check('hype_config', configs = configs)
        self._check('synchron_config', configs = configs) 
        self._check('iq_deskew_config', configs = configs)
        self._check('cd_com_config', configs = configs) 
        self._check('mimo_config', configs = configs) 
        self._check('lofo_config', configs = configs) 
        self._check('cpe_config', configs = configs) 
        self.__hype_para__()
        
        self.ldsp_obj = rx.ldsp_design.LDSP(self.hype_config, device=self.device,\
            infor_print=self.infor_print, fig_plot=self.fig_plot)

        if self._check('lpf', 1, configs):
            # brickwall, bessel, butter, gaussian, rc, rrc
            self._check('lpf_config', configs = configs) 
            self.__low_pass_filter__()
        
        if self._check('iq_balance', 1, configs):
            self._check('iq_balance_config', configs = configs)
            self.__iq_balance__()
        

        if self._check('iq_deskew', 1, configs):
            self.__iq_deskew__() 

        if self.load_len:
            self._check('len_config_path', configs = self.fiber_config)
            self.len_config = self.read_configs(self.len_config_path)
            self.len_array = np.zeros((self.span_num))
            for i in range(self.span_num):
                self.len_array[i] = self.len_config['span' + str(i+1)]
            self.total_len = np.sum(self.len_array)
        else:
            # Span length calculation, also can set different length in each span
            self.span_num = int(self.total_len / self.span_len)
            if self.span_num * self.span_len != self.total_len:
                self.span_num = self.span_num + 1
                self.len_array = np.zeros(self.span_num) + self.span_len
                self.len_array[self.span_num-1] = self.total_len - (self.span_num - 1) * self.span_len
            else:
                self.len_array = np.zeros(self.span_num) + self.span_len
            if np.sum(self.len_array) != self.total_len:
                raise ValueError('The sum of span length is different with total length !')
        
        if self._check('cdcom', 1, configs):
            if simu_configs['Simu_Para']['pre_cdcom']:
                pre_cd_len = simu_configs['Tx_Para']['pre_cdcom_config']['cd_len']
            else:
                pre_cd_len = 0
            self.__cdcom__(pre_cd_len)

        if self._check('mimo', 1, configs):
            self.__mimo__()
            
        if self._check('clofo', 1, configs):
            self.__lofo__()

        if self._check('synchronization', 1, configs):
            self._check('synchron_config', configs = configs)   # 2x2, 4x4
            self.__synchronization__()

        if self._check('cpe', 1, configs):
            self._check('cpe_config', configs = configs)   # bps, coarse
            self.__cpe__()

        if self._check('real_mimo', 1, configs):
            self._check('real_mimo_config',  configs = configs)   
            self.__real_mimo__()
        
        self.ldsp_obj.hype_init()
        self.rx_config = configs
        start_t2 = time.time() 
        self.print_buff.append('info__ rx_para initialized with {:.2f} s'.format(start_t2-start_t1))
    
    def __hype_para__(self):
        """
        
            block: 
                for i in range(block_num):
                    start_idx = padding + i*block_processed_sam
                    end_idx = start_idx + block_sam
        """

        configs = self.hype_config
        self._config_check('torch_data_type', self.torch_data_type, configs=configs)  
        self._config_check('rand_seed', self.rand_seed, configs=configs)  
        self._check("awg_memory_depth")
        self.frame_sym_num  = self._check("sym_num_per_pol")
        self.frame_num      = self._check('frame_num', configs=configs)

        "Rx sample setting"
        self.rx_sam_rate    = self._check("rx_sam_rate", configs=configs)    # GHz
        self.rx_sym_rate    = self._check("sym_rate", configs=configs)       # GHz   ns = 1/GHz
        self.rx_sam_dt      = 1000 / self.rx_sam_rate                               # ps ps = 1000 ns
        self.rx_sym_dt      = 1000 / self.rx_sym_rate
        self.rx_sam_now     = self.rx_sam_rate / self.rx_sym_rate
        self._check('rx_sam_rate', self.rx_sam_rate, configs=configs)
        self._check('rx_sym_rate', self.rx_sym_rate, configs=configs)
        self._check('rx_sam_dt', self.rx_sam_dt, configs=configs)
        self._check('rx_sym_dt', self.rx_sym_dt, configs=configs)
        self._check('rx_sam_now', self.rx_sam_now, configs=configs)

        "Batch norm setting"
        self._config_check('batch_norm', configs=configs)  

        "Chromatic dispersion compensation setting"
        cdc_upsam = self._config_check('cdc_upsam', configs=configs)  # upsam for CDC, MIMO
        cdc_sam_rate = cdc_upsam*self.rx_sym_rate
        cdc_out_sym = self._config_check('cdc_out_sym', configs=configs)  #
        if self.frame_sym_num % cdc_out_sym != 0:
            raise RuntimeError('The frame symbol number must be an integer multiple of CDC block symbol number')
        else:
            frame_block_num = int(self.frame_sym_num/cdc_out_sym)
            self._config_check('frame_block_num', frame_block_num, configs=configs) 
        cdc_padding_sym = self._config_check('cdc_padding_sym', configs=configs)  #
        if cdc_padding_sym % 2 != 0:
            raise RuntimeError('The CDC block paddings must be an integer multiple of 2')
        cdc_half_padding_sym = int(cdc_padding_sym/2)
        cdc_padding_sam = int(cdc_padding_sym*cdc_upsam)
        if cdc_out_sym*cdc_upsam % int(cdc_out_sym*cdc_upsam)!=0 or\
             cdc_padding_sym*cdc_upsam % int(cdc_padding_sym*cdc_upsam)!=0:
            raise RuntimeError('The CDC block samples must be an integer, please modify the upsam and block symbols')

        # CDC block setting
        cdc_block_num = frame_block_num * self.frame_num
        cdc_block_sym = cdc_padding_sym + cdc_out_sym
        cdc_block_processed = int(cdc_out_sym*cdc_upsam)
        cdc_block_sam = cdc_padding_sam + cdc_block_processed
        cdc_block_half_padding = int(cdc_padding_sam/2)

        cdc_lr = self._config_check('cdc_lr', configs=configs)
        self._config_add('upsam',   cdc_upsam, self.cd_com_config)
        self._config_add('sam_rate',   cdc_sam_rate, self.cd_com_config)
        self._config_add('out_sym', cdc_out_sym, self.cd_com_config)
        self._config_add('padding_sym', cdc_padding_sym, self.cd_com_config)
        self._config_add('half_padding_sym', cdc_half_padding_sym, self.cd_com_config)
        self._config_add('padding_sam', cdc_padding_sam, self.cd_com_config)
        self._config_add('block_num', cdc_block_num, self.cd_com_config)
        self._config_add('block_sym', cdc_block_sym, self.cd_com_config)
        self._config_add('block_processed', cdc_block_processed, self.cd_com_config)
        self._config_add('block_sam', cdc_block_sam, self.cd_com_config)
        self._config_add('block_half_padding', cdc_block_half_padding, self.cd_com_config)
        self._config_add('lr', cdc_lr, self.cd_com_config)

        # MIMO block setting
        mimo_upsam = cdc_upsam
        mimo_out_sym = self._config_check('mimo_out_sym', configs=configs)  #
        if cdc_out_sym % mimo_out_sym != 0:
            raise RuntimeError('The symbol numbers of CDC block must be an integer multiple of MIMO block symbol number')
        mimo_taps = self._config_check('mimo_taps', configs=configs)  #
        mimo_padding_sym = mimo_out_sym
        mimo_block_processed = mimo_padding_sym * mimo_upsam
        if mimo_block_processed > mimo_taps:
            raise RuntimeWarning('MIMO taps should be more than the out symbol num')
        elif mimo_block_processed-int(mimo_block_processed) != 0:
            raise RuntimeError('MIMO block must be integer')
        else:
            mimo_block_processed = int(mimo_block_processed)
        mimo_block_sym = mimo_out_sym + mimo_padding_sym
        mimo_block_num = int(cdc_out_sym/mimo_out_sym)
        mimo_block_padding = mimo_block_processed
        mimo_block_half_padding = int(mimo_block_padding/2)
        mimo_block_sam = mimo_block_padding+mimo_block_processed

        mimo_init_value = self._config_check('mimo_init_value', configs = configs)
        mimo_cma_lr = self._config_check('mimo_cma_lr', configs = configs)
        mimo_lms_lr = self._config_check('mimo_lms_lr', configs = configs)

        self._config_add('upsam',   mimo_upsam,     self.mimo_config)
        self._config_add('out_sym', mimo_out_sym,   self.mimo_config)
        self._config_add('taps', mimo_taps,         self.mimo_config)
        self._config_add('padding_sym', mimo_padding_sym, self.mimo_config)

        self._config_add('block_num', mimo_block_num, self.mimo_config)
        self._config_add('block_sym', mimo_block_sym, self.mimo_config)
        self._config_add('block_processed', mimo_block_processed,   self.mimo_config)
        self._config_add('block_sam', mimo_block_sam, self.mimo_config)
        self._config_add('block_padding', mimo_block_padding,       self.mimo_config)
        self._config_add('block_half_padding', mimo_block_half_padding, self.mimo_config)

        self._config_add('init_value', mimo_init_value,     self.mimo_config)
        self._config_add('cma_lr', mimo_cma_lr, self.mimo_config)
        self._config_add('lms_lr', mimo_lms_lr, self.mimo_config)

        # Oscope setting
        oscope_sam_rate = self.rx_sam_rate
        oscope_upsam = self.rx_sam_now
        self._check('oscope_sam_rate', oscope_sam_rate, configs=configs)
        self._check('oscope_upsam', oscope_upsam, configs=configs)

        # Rx setting
        rx_block_upsam = cdc_upsam
        rx_sam = int((self.frame_num*self.frame_sym_num+cdc_padding_sym)*rx_block_upsam)
        rx_block_sym = cdc_block_sym
        rx_block_sam = cdc_block_sam
        rx_block_num = int(cdc_block_num*self.frame_num)
        rx_dsam_rate = oscope_upsam / cdc_upsam
        self._check('rx_sam', rx_sam, configs=configs)
        self._check('rx_block_upsam', rx_block_upsam, configs=configs)
        self._check('rx_block_sym', rx_block_sym, configs=configs)
        self._check('rx_block_sam', rx_block_sam, configs=configs)
        self._check('rx_block_num', rx_block_num, configs=configs)
        self._check('rx_dsam_rate', rx_dsam_rate, configs=configs)

        # skew setting
        skew_lr = self._config_check('skew_lr', configs=configs)  #
        self._config_add('lr', skew_lr, self.iq_deskew_config)
        self._config_add('sam_rate', cdc_sam_rate, self.iq_deskew_config)
        self._config_add('block_sam', cdc_block_sam, self.iq_deskew_config)

        # lofo setting
        lofo_lr = self._config_check('lofo_lr', configs=configs)  #
        self._config_add('lr', lofo_lr, self.lofo_config)
        self._config_add('clofo_sam_rate', cdc_sam_rate, self.lofo_config)
        self._config_add('clofo_block_sam', cdc_block_sam, self.lofo_config)
        self._config_add('flofo_sam_rate', self.rx_sym_rate, self.lofo_config)
        self._config_add('flofo_block_sam', mimo_block_sym, self.lofo_config)

        
        # cpe setting
        cpe_lr = self._config_check('cpe_lr', configs=configs)  #
        self._config_add('lr', cpe_lr, self.cpe_config)
        self._config_add('sam_rate', self.rx_sym_rate, self.cpe_config)
        self._config_add('block_sam', mimo_block_sym, self.cpe_config)
        self._config_add('ccpe_sam', mimo_block_sam, self.cpe_config)



    def __iq_balance__(self):
        mode = self.iq_balance_config['type']
        self.iq_balance_obj = rx.imbalance_eq.IQ_Balance(mode, data_mode = 'torch', device = self.device)

    def __iq_deskew__(self):
        config = self.iq_deskew_config
        self._config_check('mode', 'freq', config)   
        self._config_check('data_mode', 'numpy', config)   
        self._config_check('device', self.device, config)   
        self.iq_deskew_obj = rx.skew_eq.Skew_Equalization(**config)
        if config['mode'] == 'freq':
            config['args'] = {'sam_rate':self.rx_sam_rate,
                'skew_list':config['skew_list']}
        self.iq_deskew_obj.init(**config['args'])

        self._config_check('skew_list', configs=config) 
        self.ldsp_obj.iq_deskew_init(config)

    def __low_pass_filter__(self):
        # brickwall, bessel, butter, gaussian, rc, rrc
        config = self.lpf_config
        self._config_check('mode', 'freq', config)   
        self._config_check('sam_rate', self.rx_sam_rate, config)   
        self._config_check('fft_num', self.oscop_sam_num, config)   
        self._config_check('data_mode', 'torch', config)   
        self._config_check('device', self.device, config)  
        self.lpf_obj = rx.lpf_design.LPF(**config)

        lpf_args = config['args']
        max_lofo = self._config_check('max_lofo', 0.5, lpf_args)
        roll_off = self._config_check('roll_off', 0.1, lpf_args)
        filter_bw = self._config_check('filter_bw',\
            self.sym_rate * (1 + roll_off) / 2 + max_lofo,lpf_args)
        if self.lpf_obj.mode == 'rrc':
            self._config_check('beta', roll_off, lpf_args)
            self._config_check('upsam', self.sam_now, lpf_args)
            if max_lofo != 0:
                self.print_buff.append('warning__    rx_para LPF:RRC will filter the signal becasue of LOFO at rx_para')
                # self.logger.warn('RRC will filter the signal becasue of LOFO at rx_para')
        else:
            if filter_bw > self.channel_space:
                filter_bw = self.channel_space
            if self.lpf_obj.mode == 'bessel':
                self._config_check('order', 15, lpf_args)     # bessel para
            elif self.lpf_obj.mode == 'butter':
                self._config_check('gpass', 5, lpf_args)     # butterworth para
                self._config_check('gstop', 40, lpf_args)    # butterworth para
            self._config_check('cut_off', filter_bw, lpf_args)
            if self.infor_print:
                # self.logger.info("    rx_para LPF: {} filter and bandwidth {:.2f} GHz".format(config['mode'],filter_bw))
                self.print_buff.append('info__    rx_para LPF: {} filter and bandwidth {:.2f} GHz'.format(config['mode'],filter_bw))
        self._config_check('comp_s21', 1, lpf_args)
        self._config_check('s21_bw', filter_bw, lpf_args)
        self._config_check('s21_path', configs=lpf_args)
        self.lpf_obj.init(**lpf_args)

    def __nolinearity_compensation__(self):
        if self.nl_com_config['type'] == 'DBP':
            alpha_loss = self.fiber_config['alpha_loss'] 
            gamma = self.fiber_config['gamma'] 
            sym_sam = self._config_check('sym_sam', 2, self.nl_com_config['args'])    
            self._config_check('alpha_loss', alpha_loss, self.nl_com_config['args'])    
            self._config_check('gamma', gamma, self.nl_com_config['args'])    
            self.cdcom = 0
            self.fft_num = self.sym_num_per_pol * sym_sam
            self.dt = self.sym_time / sym_sam
            if self.rx_data_mode == 'torch':
                self.w = torch.from_numpy(self.w).to(self.device)
            self.w = 2 *self.Constant_pi * np.arange(- int(self.fft_num / 2), int(self.fft_num / 2)) / (self.dt * (self.fft_num - 1))
            self.phase_factor_freq = self.beta0[self.cut_idx]\
                + self.beta1[self.cut_idx] * (self.w)\
                    + self.beta2[self.cut_idx] * (self.w ** 2) / 2\
                        + self.beta3 * (self.w ** 3) / 6  

    def __cdcom__(self, pre_cd_len):
        wavelength = self._check('wavelength', 1550, self.fiber_config) # wavelength (nm)=10^-9(m)
        # Dispersion parameters
        D = self._config_check('D', 17, self.fiber_config) # 16.75  dispersion Parameter (ps/nm.km); if anomalous dispersion(for compensation),D is negative
        S = self._config_check('S', 0.075, self.fiber_config) # S slope ps/(nm^2.km)
        beta2c = self._config_check('beta2c',\
            - 1000 * D * wavelength ** 2 / (2 * self.Constant_pi * self.Constant_C), self.fiber_config) # beta2 (ps^2/km);
        beta3c = self._config_check('beta3c', 0, self.fiber_config)
        self.df = ((np.arange(1, self.channel_num + 1) - (self.channel_num + 1) / 2) * self.channel_space)
        self.beta0 = beta2c / 2 * (2 * self.Constant_pi * self.df)**2 * 10**-6 + beta3c / 6 * (2 * self.Constant_pi * self.df)**3 * 10**-9   #WDM的接收端的cdc与df有关，所以要更新参数
        self.beta1 = beta2c * 2 * self.Constant_pi * self.df * 10**-3 + beta3c * (2 * self.Constant_pi * self.df)**2 * 10**-6 / 2      
        self.beta2 = beta2c + 2 * self.Constant_pi * self.df * beta3c * 10**-3
        self.beta3 = beta3c
        self.sig_bw = self.rx_sym_rate * (1+0.1)/2
        config = self.cd_com_config
        beta = [self.beta0, self.beta1, self.beta2, self.beta3]
        cdc_length = self._config_check('cdc_length', 'auto', config) 
        if cdc_length == 'auto':
            cdc_length = self.total_len - pre_cd_len
            config['cdc_length'] = cdc_length
        else:
            cdc_length = self._config_check('cdc_length', self.total_len, config)
        self.cdc_isi_sym = calcu.ISI_induced_by_dispersion(self.sig_bw, self.rx_sym_dt,\
             D, wavelength, self.total_len)
        self.cdc_w = 2*self.Constant_pi*calcu.digital_freq_wshift(config['block_sam'],\
             config['sam_rate']*1e-3, data_mode='torch', device=self.device) 
        self.cdc_phase_factor_freq = beta[0][self.cut_idx]\
                + beta[1][self.cut_idx] * (self.cdc_w)\
                    + beta[2][self.cut_idx] * (self.cdc_w ** 2) / 2\
                        + beta[3] * (self.cdc_w ** 3) / 6
        self.cdc_tap = torch.exp(1j * self.cdc_phase_factor_freq * cdc_length)
        self.cdc_tap = torch.stack((self.cdc_tap,self.cdc_tap), dim=0)
        self._config_add('isi_sym', self.cdc_isi_sym, config)
        self._config_add('cdc_tap', self.cdc_tap, config)
        
        self.ldsp_obj.cdc_init(config)

    def __synchronization__(self):
        config = self.synchron_config
        self._config_check('mode', config)   
        self._config_check('corr_num', config)   
        self.ldsp_obj.synchron_init(config)
     
    def __mimo__(self):

        config = self.mimo_config
        mode = self._config_check('mode', configs=config)   
        self._config_check('radius_idx', configs=config)   
        in_dim, out_dim = mode.split('x')
        in_dim, out_dim = int(in_dim), int(out_dim)
        in_sam_dim = config['block_sam']
        td_taps = torch.zeros((in_dim, out_dim, config['block_processed']), device=self.device) + 0j
        middle_idx = int(config['block_processed']/2)
        # mimo init
        if in_dim == 2:
            td_taps[0,0,0] += config['init_value'] 
            td_taps[1,1,0] += config['init_value'] 
        elif in_dim == 4:
            td_taps[0,0,0] += config['init_value'] 
            td_taps[1,0,0] += config['init_value'] 
            td_taps[2,1,0] += config['init_value'] 
            td_taps[3,1,0] += config['init_value'] 
        
        fd_taps = fftshift(fft(td_taps, n=in_sam_dim, dim=-1), dim=-1)

        self._config_add('td_taps', td_taps, configs=config)   
        self._config_add('fd_taps', fd_taps, configs=config)   
        self.ldsp_obj.mimo_init(config)

    def __lofo__(self):

        config = self.lofo_config
        lofo_locked = 0
        self._config_check('mode', configs=config)  
        self._config_add('lofo_est', np.zeros((1)), configs=config)   
        self._config_add('lofo_locked', lofo_locked, configs=config)   
        self.ldsp_obj.lofo_init(config)
    
    def __cpe__(self):
        config = self.cpe_config
        self._config_check('mode', configs=config)   
        self._config_check('test_num', configs=config) 
        self._config_check('ml', configs=config)
        self._config_check('data_aided', configs=config)   
        self.ldsp_obj.cpe_init(config)


    def __real_mimo__(self):
        self._config_check('mode', 'TD_4x4', self.real_mimo_config)   
        self._config_check('lr_optim', 'constant', self.real_mimo_config)   
        self._config_check('half_taps_num', 15, self.real_mimo_config)   
        self._config_check('out_sym_num', 1, self.real_mimo_config)   
        self._config_check('upsam', 1, self.real_mimo_config)   
        self._config_check('block_size', 1, self.real_mimo_config)   
        self._config_check('data_mode', 'numpy', self.real_mimo_config)   
        self._config_check('device', self.device, self.real_mimo_config)   
        self._config_check('infor_print', self.infor_print, self.real_mimo_config)   
        self.real_mimo_obj = rx.adaptive_filter_design.Vanilla(**self.real_mimo_config)
        
        mimo_args = self.real_mimo_config['args']
        upsam = self._config_check('upsam', 1, mimo_args) 
        sam_num = self.frame_sym_num * upsam
        self._config_check('algo_type', 'lms', mimo_args)  # cma, mma, dd_lms
        self._config_check('cma_pretrain', 0, mimo_args)  
        self._config_check('pre_train_iter', 0, mimo_args)  
        self._config_check('train_num', 65536, mimo_args)  
        self._config_check('train_epoch', 2, mimo_args)  
        self._config_check('lr', 5.0e-4, mimo_args)  
        self._config_check('sam_num', sam_num, mimo_args) 
        self._config_check('tap_init_mode', 1, mimo_args) 
        self._config_check('tap_init_value', 1.0, mimo_args) 
        self.real_mimo_obj.init(**mimo_args)
