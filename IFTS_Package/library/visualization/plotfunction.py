import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
#from regex import D
import seaborn as sns
import torch
#from IFTS_Package.module_para.signal_para import Sig_Para
from IFTS import visualization as visual 

                                #函数说明

#使用前需要导入 from IFTS_Package.plotpicture import plotplot 
#这个函数库和sig_para类解耦，但如scatter_plot_npol函数需要在调用时传入front_sym_num，
# constellation_points， integerseq，path参数（这些参数也可直接通过sig_para.XXX来传入），
#用**keyword字典来接收相关参数
#(get color 等函数包含在新的scatter_plot_npol函数里，需传入integerseq)
# 另外字体大小和字体样式可选择性传入，默认字体大小size=16，字体类型family=times new roman
      
 
#family:字体类型；常用的有simsun(宋体)，times new roman(新罗),SimHei(黑体）；

scatter_colour = 'gold'

def data_mode_check(x):   
        x_type = type(x)
        if x_type is torch.Tensor:
            return x.clone().detach().cpu().numpy()
        else:
            return x

def get_max_value(x):
        if type(x) == np.ndarray:
            return max([np.max(np.abs(x.real)),\
                 np.max(np.abs(x.imag))])
        else:
            return max([torch.max((x.real).abs()).item(),\
                torch.max((x.imag).abs()).item()])

def get_idx(keyword, sps):
        return (keyword.get('front_sym_num')+np.arange(keyword.get('constellation_points'))) * sps

def get_colour(x, keyword, sps = 1):
        idx = get_idx(keyword, sps)
        nPol = len(x) 
        c = []
        colour = np.array(sns.color_palette("husl", keyword.get('class_num')))
        for i_p in range(nPol):
            c.append(colour[x[i_p][idx]])
        colour_seq = c
        colour_index = np.arange(idx.shape[0])
    
        #print(c)
        #print(colour_seq)
        #print(colour_index)
        return colour_seq, colour_index

#scatter_plot_npol函数需要传入front_sym_num，constellation_points， integerse，path参数，字体大小和字体样式可选择性传入
def scatter_plot_nPol(x, sps = 1, name = 'Constellation', set_c = 0,\
        xlim = None, ylim = None, s = None, discard_sam = None, fontfamily = 'Times New Roman', fontsize = '16' ,**keyword):
        #sns.set_theme(style="whitegrid")
        plt.rcParams['font.family'] = fontfamily
        plt.rcParams['font.size'] = fontsize
        

        if not set_c:
         colour_seq, colour_index = get_colour(keyword.get('integerseq') ,keyword, sps=1)
        nPol = len(x)
        pol_name = ["X Polarization", "Y Polarization"]
        #fig = plt.figure(figsize=(6.5 * nPol, 6.5))
        #gs = gridspec.GridSpec(1, nPol, figure=fig, wspace=0.5)
        fig, axs = plt.subplots(1, nPol, figsize=(6.5* nPol, 6.5), edgecolor = '#282C34')
        fig.set_facecolor('#FFFFFF')
        #fig.set_edgecolor('#000000')
        #fig.suptitle(name, color = '#B4B6BD', fontsize = 'xx-large')
        fig.suptitle(name, color = '#000000', fontsize = 'xx-large')
        # fig, axs = plt.subplots(1, nPol, figsize=(6.5* nPol, 6.5))
        # fig.suptitle(name, fontsize = 'xx-large')
        if xlim == None or ylim == None:
            xy_max = np.max(np.array([get_max_value(x[i_p]) for i_p in range(nPol)]))
            xlim = [-xy_max, xy_max]
            ylim = [-xy_max, xy_max]
        idx = get_idx(keyword, sps=sps)
        if discard_sam is not None:
            plot_num = sps*(int(discard_sam)+2*keyword.get('front_sym_num')+keyword.get('constellation_points'))
            if len(x[0]) < plot_num:
                discard_sam = len(x[0])-plot_num 
            idx = idx + int(discard_sam)
            set_c = 1
        for i_p in range(nPol):
            y = x[i_p][idx]
            y = data_mode_check(y)
            if set_c:
                c = scatter_colour
            else:
                if len(colour_seq) == 0:
                    c = None
                else:
                    colour_index = np.arange(idx.shape[0])
                    c = colour_seq[i_p][colour_index]
            # axs[i_p] = plt_con.scatter_plot_white(y, axs[i_p], pol_name[i_p], c, xlim, ylim, s)
            axs[i_p] = visual.plt_con.scatter_plot_black_plotfunction(y, axs[i_p], pol_name[i_p], c, xlim, ylim, s)
        #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=None)
        fig.savefig(keyword.get('path') + name + '.png', dpi = 500)
        plt.close('all')


def sym_map_plot(x, name = 'Constellation', set_c = 1, xlim = None, ylim = None, s = None, c_index = None, fontfamily = 'Times New Roman', fontsize = '16' ,**keyword):
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['font.family'] = fontfamily
        fig, axs = plt.subplots(1, 1, figsize=(6.5, 6.5))
        y = data_mode_check(x)
        if set_c:
            c = np.array(sns.color_palette("husl", keyword.get('class_num')))
            #print(keyword.get('class_num'))
        else:
            c = scatter_colour 
        axs = visual.plt_con.scatter_plot(y, axs, 'sym_map', c, xlim, ylim, s)
        fig.savefig(keyword.get('path') + name + '.png', dpi = 900)
        plt.close('all')

def scatter_plot(x, sam_num = 1, name = 'Constellation', set_c = 0, xlim = None, ylim = None, s = None, c_index = None, fontfamily = 'Times New Roman', fontsize = '16', **keyword):
        nPol = len(x)
        plt.rcParams['font.family'] = fontfamily
        plt.rcParams['font.size'] = fontsize
        if set_c:
         colour_seq, colour_index = get_colour(keyword.get('integerseq'), sps=1)
        pol_name = ["X Polarization"]
        fig, axs = plt.subplots(1, 1, figsize=(6.5, 6.5))
        # fig.suptitle(name, color = '#B4B6BD', fontsize = 'xx-large')
        idx = get_idx(sam_num = sam_num)
        # for i_p in range(nPol):
        y = x[0][idx]
        y = data_mode_check(y)
        if set_c:
            c = scatter_colour
        else:
            if  len(colour_seq) == 0:
                c = None
            else:
                c = colour_seq[0][colour_index]
            axs = visual.plt_con.scatter_plot(y, axs, pol_name[0], c, xlim, ylim, s)

        fig.savefig(keyword.get('path') + name + '.png', dpi = 500)
        plt.close('all')
    
def scatter_heat_map_nPol(x, sam_num = 1, name = 'Constellation', xlim = None, ylim = None, s = None, fontfamily = 'Times New Roman', fontsize = '16', **keyword):
        nPol = len(x)
        plt.rcParams['font.family'] = fontfamily
        plt.rcParams['font.size'] = fontsize
        pol_name = ["X Polarization", "Y Polarization"]
        fig, axs = plt.subplots(1, nPol, figsize=(6.5* nPol, 6.5), edgecolor = '#282C34')
        fig.set_facecolor('#282C34')
        fig.set_edgecolor('#282C34')
        fig.suptitle(name, color = '#B4B6BD', fontsize = 'xx-large')
        idx = get_idx(sam_num = sam_num)
        for i_p in range(nPol):
            y = x[i_p][idx]
            y = data_mode_check(y)
            axs[i_p] = visual.plt_con.scatter_heat_map(y, axs[i_p], pol_name[i_p], xlim, ylim, s)

        fig.savefig(keyword.get('path') + name + '.png', dpi = 500) 
        plt.close('all')

def loss_plot_nPol(loss, plot_space = 10, smooth = 1, ema_span = 40, name = 'Loss function', fontfamily = 'Times New Roman', fontsize = '16', **keyword):
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['font.family'] = fontfamily
        out_dim = loss.shape[-1]
        if out_dim == 2:
            subname = ['loss_x', 'loss_y']
        elif out_dim == 4:
            subname = ['loss_xi', 'loss_xq', 'loss_yi', 'loss_yq']
       # loss = self.data_mode_check(loss)
        loss = data_mode_check(loss)
        max_value = np.amax(loss)
        min_value = np.amin(loss)
        fig, axs = plt.subplots(1, out_dim, figsize=(8 * out_dim, 6.5))
        fig.suptitle(name, fontsize = 'xx-large')
       # loss = self.data_mode_check(loss)
        loss = data_mode_check(loss)
        for i in range(out_dim):
            axs[i] = visual.plt_wav.wave1d_plot(np.abs(loss[::plot_space, i]), axs[i], \
                title = subname[i], smooth=smooth, EMA_SPAN = ema_span)
            axs[i].set_ylim(max(min_value-0.05, 0), max_value) 
            axs[i].tick_params(axis='x', labelsize=20)  # X轴标签字体大小
            axs[i].tick_params(axis='y', labelsize=20)  # Y轴标签字体大小   
        fig.savefig(keyword.get('path')+ name + '.png' , dpi = 500)
        plt.close('all')


def psd(x, fc=0, hz = 'GHz', name = 'power_density', xlim=None, ylim=None, fontfamily = 'Times New Roman', fontsize = '16',  **keyword):
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['font.family'] = fontfamily
        x = data_mode_check(x)
        fig, axs = plt.subplots(1, figsize=(6.5, 6.7))
        fig.suptitle(name, fontsize = 'x-large')
        if hz == 'GHz':
            sam_rate = keyword.get('sam_rate')
            label = 'Frequency(GHz)'
        else:
            sam_rate = keyword.get('sam_rate') * 10 ** -3
            label = 'Frequency(THz)'  
        psd, f = axs.psd(x, 4096, Fs = sam_rate, Fc = fc)
        psd_log = 10 * np.log10(psd)
        axs.grid(True)
        axs.set_ylabel('')
        axs.set_xlabel(label,fontsize = 'large')#修改Frequency字体大小
        if xlim is not None:
            axs.set_xlim(xlim)
        if ylim is not None:
            axs.set_ylim(ylim)
        else:
            axs.set_ylim([np.max(psd_log) - 60, np.max(psd_log)+1])
        axs.axvline(x=fc, ls='--', color='r')
        axs.text(x=fc+0.01, y = np.max(psd_log)-58,\
            s = str(round(fc,2))+' THz', color='r')
        axs.tick_params(axis='x', labelsize=16)  # X轴标签字体大小
        axs.tick_params(axis='y', labelsize=16)  # Y轴标签字体大小
        fig.savefig(keyword.get('path') + name + '.png', dpi = 900) 
        plt.close('all')
    
def psd_link( x, start, end, now, fc=0, hz = 'GHz', name = 'power_density', norm=0, xlim=None, ylim=None, fig=None, axs=None, fontfamily = 'Times New Roman', fontsize = '16',  **keyword):
        if hz == 'GHz':
            sam_rate = keyword.get('sam_rate')
            label = 'Frequency(GHz)'
        else:
            sam_rate = keyword.get('sam_rate') * 10 ** -3
            label = 'Frequency(THz)' 
        if now == start:
            sns.set_theme(style="darkgrid")#set_theme内置函数会自动重新修改字体大小为16
            plt.rcParams['font.size'] = fontsize#字体大小
            plt.rcParams['font.family'] = fontfamily
            fig, axs = plt.subplots(1, figsize=(6.5, 6.7))
            fig.suptitle(name, fontsize = 'x-large')
        x = data_mode_check(x)
        fig0, axs0 = plt.subplots()
        psd, f = axs0.psd(x, 4096, Fs = sam_rate, Fc = fc)
        plt.close(fig=fig0)
        if norm:
            psd = psd/np.max(psd)
        psd_log = 10 * np.log10(psd)
        axs.plot(f, psd_log)
        if end == now:
            axs.grid(True)
            axs.set_ylabel('')
            axs.set_xlabel(label,fontsize = 'large')#x轴坐标标题大小
            if xlim is not None:
                axs.set_xlim(xlim)
            if ylim is not None:
                axs.set_ylim(ylim)
            else:
                axs.set_ylim([np.max(psd_log) - 35, np.max(psd_log)+1])
            axs.axvline(x=fc, ls='--', color='r')
            axs.text(x=fc+0.005, y = np.max(psd_log)-5,\
                s = str(round(fc,2))+' THz', color='r')
            axs.tick_params(axis='x', labelsize=16)  # X轴标签字体大小
            axs.tick_params(axis='y', labelsize=16)  # Y轴标签字体大小
            fig.savefig(keyword.get('path') + name + '.png', dpi = 500) 
            plt.close('all')
        return fig, axs
    
def psd_nPol(x, fc=0, hz = 'GHz', name = 'power_density', sign = None,fontfamily = 'Times New Roman', fontsize = '16', **keyword):
        nPol = len(x)
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['font.family'] = fontfamily
        pol_name = ["X Polarization", "Y Polarization"]
        fig, axs = plt.subplots(1, nPol, figsize=(6.5* nPol, 6.5))
        fig.suptitle(name, fontsize = 'xx-large')
        for i_p in range(nPol):
            y = x[i_p]
            y = data_mode_check(y)
            if hz == 'GHz':
                sam_rate = keyword.get('sam_rate')
                label = 'Frequency(GHz)'
            else:
                sam_rate = keyword.get('sam_rate') * 10 ** -3
                label = 'Frequency(THz)'  
            psd, f = axs[i_p].psd(y, 4096, Fs = sam_rate, Fc = fc)
            psd_log = 10 * np.log10(psd)
            axs[i_p].grid(True)
            axs[i_p].set_ylabel('')
            axs[i_p].set_xlabel(label, fontsize = 'large')
            if sign is not None:
                axs[i_p].axhline(y=np.max(psd_log)-10)
                axs[i_p].axvline(x=sign)
                axs[i_p].text(x = sign-5, y = np.max(psd_log)-9, s = '(' + str(np.around(sign, 1)) +' GHz, -10 dB)')
                axs[i_p].set_ylim([np.max(psd_log) - 60, np.max(psd_log)+1])
                axs[i_p].set_xlim([-sign*1.5, sign*1.5])
            axs[i_p].set_title(pol_name[i_p], fontsize = 'x-large')
            axs[i_p].tick_params(axis='x', labelsize=16)  # X轴标签字体大小
            axs[i_p].tick_params(axis='y', labelsize=16)  # Y轴标签字体大小
        fig.savefig(keyword.get('path') + name + '.png', dpi = 500) 
        plt.close('all')
def err_hist_plot_nPol(sym_map, para,fontfamily = 'Times New Roman', fontsize = '16', **keyword):
        plt.rcParams['font.family'] = fontfamily
        plt.rcParams['font.size'] = fontsize
        nPol = para.nPol
        pol_name = ["X", "Y"]
        for i_p in range(nPol):
            visual.plt_analysis.err_analysis_plot(para.sym_err[i_p], para.sym_err_idx[i_p], sym_map, name = keyword.get('path') + pol_name[i_p] + 'hist')
        plt.close('all')
def err_analysis_plot(rx_sym, tx_sym_idx, err_sym, err_sym_idx, sym_map, i_p, fontfamily = 'Times New Roman', fontsize = '16',**keyword):
        plt.rcParams['font.family'] = fontfamily
        plt.rcParams['font.size'] = fontsize
        pol_name = ["X", "Y"]
        if err_sym_idx.shape[0] >= 256:
            visual.plt_analysis.err_analysis_plot_v2(rx_sym, tx_sym_idx, err_sym, err_sym_idx, sym_map, name = keyword.get('path') + pol_name[i_p] + 'hist')    
        else:
            print('Few symbol errors of ' + pol_name[i_p] + ' are not allowed to plot error histgram here')
            # SNR of each constellation points are allowed only
            visual.plt_analysis.err_analysis_plot_v2(rx_sym, tx_sym_idx, err_sym, err_sym_idx, sym_map, hist = 0, name = keyword.get('path') + pol_name[i_p] + 'hist')    
        plt.close('all')   

def firtap_plot_nPol_old(mimo_obj, name = 'Firtap', fontfamily = 'Times New Roman', fontsize = '16',**keyword):
        if mimo_obj.fir_type == '2x2':
            plot_num = 4
            subname = ['hxx', 'hxy', 'hyx', 'hyy']
            if mimo_obj.mat_op:
                x = mimo_obj.h.transpose((2, 0, 1)).reshape((plot_num, -1))
            else:
                x = [mimo_obj.hxx, mimo_obj.hxy,\
                    mimo_obj.hyx, mimo_obj.hyy]
            fig, axs = plt.subplots(2, 2, figsize=(6.5* 2, 6.5 * 2))
        elif mimo_obj.fir_type == '4x2':
            plot_num = 8
            if mimo_obj.mat_op:
                x = mimo_obj.h.transpose((2, 0, 1)).reshape((plot_num, -1))
            else:
                x = [mimo_obj.hx_xi, mimo_obj.hx_xq, mimo_obj.hx_yi, mimo_obj.hx_yq,\
                    mimo_obj.hy_xi, mimo_obj.hy_xq, mimo_obj.hy_yi, mimo_obj.hy_yq]
            subname = ['hx_xi', 'hx_xq', 'hx_yi', 'hx_yq', \
            'hy_xi', 'hy_xq', 'hy_yi', 'hy_yq']
            fig, axs = plt.subplots(2, 4, figsize=(6.5* 4, 6.5 * 2))
        
        sns.set_theme(style="darkgrid")
        axs = axs.reshape(-1)
        fig.suptitle(name, fontsize = 'xx-large')
        for i in range(plot_num):
            axs[i] = visual.plt_wav.wave1d_plot(np.real(x[i]), axs[i], title = subname[i], smooth=0)
            axs[i] = visual.plt_wav.wave1d_plot(np.imag(x[i]), axs[i], title = subname[i], smooth=0)
        fig.savefig(keyword.get('path') + name + '.png', dpi = 500)
        plt.close('all')
def firtap_plot_nPol(mimo_obj, name = 'Firtap', fontfamily = 'Times New Roman', fontsize = '16',**keyword):
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['font.family'] = fontfamily
        if mimo_obj.fir_type == '2x2':
            plot_num = 4
            subname = ['hxx', 'hxy', 'hyx', 'hyy']
            fig, axs = plt.subplots(2, 2, figsize=(6.5* 2, 6.5 * 2))
            for ax in axs.flat:
             ax.tick_params(axis='x', labelsize=20) 
             ax.tick_params(axis='y', labelsize=20)  # X轴标签字体大小
        elif mimo_obj.fir_type == '4x2':
            plot_num = 8
            subname = ['hx_xi', 'hx_xq', 'hx_yi', 'hx_yq', \
            'hy_xi', 'hy_xq', 'hy_yi', 'hy_yq']
            fig, axs = plt.subplots(2, 4, figsize=(6.5* 4, 6.5 * 2))
        elif mimo_obj.fir_type == '4x4':
            plot_num = 16
            subname = ['hxi_xi', 'hxi_xq', 'hxi_yi', 'hxi_yq', \
            'hxq_xi', 'hxq_xq', 'hxq_yi', 'hxq_yq', \
            'hyi_xi', 'hyi_xq', 'hyi_yi', 'hyi_yq',\
            'hyq_xi', 'hyq_xq', 'hyq_yi', 'hyq_yq']
            fig, axs = plt.subplots(4, 4, figsize=(6.5* 4, 6.5 * 4))
        x = mimo_obj.h[..., 0].reshape((plot_num, -1))
        x = data_mode_check(x)
        
        sns.set_theme(style="darkgrid")
        plt.rcParams['font.size'] = '16'
        plt.rcParams['font.family'] = fontfamily
        axs = axs.reshape(-1)
        fig.suptitle(name, fontsize = 'xx-large')
        for i in range(plot_num):
            axs[i] = visual.plt_wav.wave1d_plot(np.real(x[i]), axs[i], title = subname[i], smooth=0)
            axs[i] = visual.plt_wav.wave1d_plot(np.imag(x[i]), axs[i], title = subname[i], smooth=0)
        fig.savefig(keyword.get('path') + name + '.png', dpi = 500) 
        plt.close('all')

def wave_plot_nPol(x, name = 'Wave', fontfamily = 'Times New Roman', fontsize = '16',**keyword):
        plot_num = len(x)
        sns.set_theme(style="darkgrid")
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['font.family'] = fontfamily
        subname = ['x', 'y']
        fig, axs = plt.subplots(1, plot_num, figsize=(6.5* plot_num, 6.5))
        fig.suptitle(name, fontsize = 'xx-large')
        for i in range(plot_num):
            y = data_mode_check(x[i])
            axs[i] = visual.plt_wav.wave1d_plot(y, axs[i], title = subname[i], smooth=0)
           
        fig.savefig(keyword.get('path') + name + '.png') 
        plt.close('all')

def corr_plot(x, name = 'Symchronization', fontfamily = 'Times New Roman', fontsize = '16',**keyword):
        print('corrplot')
        row_num = x.shape[0]
        column_num = x.shape[1]
        if row_num == 2:
            subname = ['corr_xx', 'corr_xy', 'corr_yx', 'corr_yy']

        elif row_num == 4:
            subname = ['corr_xrxr', 'corr_xrxi', 'corr_xryr', 'corr_xryi',\
                        'corr_xixr', 'corr_xixi', 'corr_xiyr', 'corr_xiyi',\
                            'corr_yrxr', 'corr_yrxi', 'corr_yryr', 'corr_yryi',\
                                'corr_yixr', 'corr_yixi', 'corr_yiyr', 'corr_yiyi']

        fig, axs = plt.subplots(row_num, column_num, figsize=(5* row_num, 5 * column_num))
        sns.set_theme(style="darkgrid")
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['font.family'] = fontfamily
        fig.suptitle(name, fontsize = 'xx-large')
        for i in range(row_num):
            for j in range(column_num):
                axs[i,j] = visual.plt_wav.wave1d_plot(x[i,j], axs[i,j], title = subname[i*column_num + j], smooth=0)
                axs[i,j].set_xticks([])

        fig.savefig(keyword.get('path') + name + '.png', bbox_inches = 'tight') 
        plt.close('all')

def snr_plot(x, name = 'SNR_Movmean', fontfamily = 'Times New Roman', fontsize = '16',**keyword):
        plot_num = len(x)
        x = np.array(x)
        x = data_mode_check(x)
        sns.set_theme(style="darkgrid")
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['font.family'] = fontfamily
        fig = plt.figure(tight_layout=True, figsize=(12, 6))
        fig.suptitle(name ,fontsize = 'x-large')
        gs = gridspec.GridSpec(2, 2)        
        ax = fig.add_subplot(gs[:, 0])
        MAX_TEMP = x.mean() + 2
        MIN_TEMP = x.mean() - 2
        sns.heatmap(np.array(x).reshape((2, -1)), yticklabels=['X', 'Y'], xticklabels=False,\
            vmax=MAX_TEMP, vmin = MIN_TEMP, cmap="rocket", ax=ax)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)#调整热力图字体大小
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)
        for i in range(plot_num):
            ax = fig.add_subplot(gs[i, 1])
            ax = visual.plt_wav.wave1d_plot(x[i], ax, title = None, smooth=1, EMA_SPAN=512)
        fig.savefig(keyword.get('path') + name + '.png') 
        plt.close('all')


def frame_plot( frame, save_path=None, data_show=0, name='total_frame', fontfamily = 'Times New Roman', fontsize = '16',**keyword):
        if save_path is None:
            save_path = keyword.get('path')
        ylim = np.max(frame) + 15
        frame_max = frame.max(axis=-1)
        plt.figure(figsize=(10,10))
        sns.set_theme(style="darkgrid") 
        plt.rcParams['font.family'] = fontfamily
        plt.rcParams['font.size'] = fontsize       
        plt.subplot(2,2,1)
        if data_show:
            plt.plot(frame[0,0], label='xx _max{:.2f}'.format(frame_max[0,0]))
            plt.plot(frame[0,1], label='xx*_max{:.2f}'.format(frame_max[0,1]))
        else:
            plt.plot(frame[0,0], label='xx')
            plt.plot(frame[0,1], label='xx*') 
        plt.ylim([0, ylim])
        plt.title('xx')
        plt.legend()
        plt.subplot(2,2,2)
        if data_show:
            plt.plot(frame[0,2], label='xy _max{:.2f}'.format(frame_max[0,2]))
            plt.plot(frame[0,3], label='xy*_max{:.2f}'.format(frame_max[0,3]))
        else:
            plt.plot(frame[0,2], label='xy')
            plt.plot(frame[0,3], label='xy*')
        plt.ylim([0, ylim])
        plt.title('xy')
        plt.legend()
        plt.subplot(2,2,3)
        if data_show:
            plt.plot(frame[1,0], label='yx _max{:.2f}'.format(frame_max[1,0]))
            plt.plot(frame[1,1], label='yx*_max{:.2f}'.format(frame_max[1,1]))
        else:
            plt.plot(frame[1,0], label='yx')
            plt.plot(frame[1,1], label='yx*')
        plt.ylim([0, ylim])
        plt.title('yx')
        plt.legend()
        plt.subplot(2,2,4)
        if data_show:
            plt.plot(frame[1,2], label='yy _max{:.2f}'.format(frame_max[1,2]))
            plt.plot(frame[1,3], label='yy*_max{:.2f}'.format(frame_max[1,3]))
        else:
            plt.plot(frame[1,2], label='yy')
            plt.plot(frame[1,3], label='yy*')
        plt.ylim([0, ylim])
        plt.title('yy')
        plt.legend()
        plt.savefig(save_path+name+'.png', dpi=500)

        plt.close('all')


def ldsp_plot(sig_out, mimo_td_window, mimo_fd, bn,\
         cdc_tap, lofo, skew, mimo_skew=None, frame=None, name='Train', set_c=True,fontfamily = 'Times New Roman', fontsize = '16', **keyword):
        # sig out plot
        print('ldsp')
        mimo_td_window = mimo_td_window.detach().cpu().numpy()
        mimo_fd = mimo_fd.detach().cpu().numpy()
        mimo_fd_abs = np.abs(mimo_fd)
        mimo_fd_phase = np.unwrap(np.angle(mimo_fd), axis=-1)
        mimo_td = np.fft.ifft(np.fft.ifftshift(mimo_fd, axes=-1), axis=-1)
        mimo_td = np.roll(mimo_td, shift=int(mimo_fd.shape[-1]/4))[..., 0:int(mimo_fd.shape[-1]/2)]
        cdc_tap = cdc_tap.detach().cpu().numpy()
        cdc_tap_phase = np.unwrap(np.angle(cdc_tap), axis=-1)
        lofo = np.array(lofo)
        skew = np.array(skew)
        save_path = keyword.get('path')+name+'_'
        if sig_out.shape[-1] > 105000:
            scatter_plot_nPol(sig_out[:, -105000:], sps=1, name=name+'_sig_out', set_c=set_c,\
                xlim=None, ylim=None, s=4, discard_sam=None, class_num = keyword.get('class_num'), \
                    constellation_points = keyword.get('constellation_points'), front_sym_num = keyword.get('front_sym_num'),\
                        path = keyword.get('path'), integerseq = keyword.get('integerseq'))
        else:
            scatter_plot_nPol(sig_out, sps=1, name=name+'_sig_out', set_c=set_c,\
                xlim=None, ylim=None, s=4, discard_sam=None, class_num = keyword.get('class_num'), \
                    constellation_points = keyword.get('constellation_points'), front_sym_num = keyword.get('front_sym_num'),\
                        path = keyword.get('path'), integerseq = keyword.get('integerseq'))
        sns.set_theme(style="darkgrid")     
        if bn is not None:   
            if len(bn) == 4:
                bn1_var = np.array(bn[1])
                bn2_var = np.array(bn[3])
                bias1, bias2 = bn[0], bn[2]
                bn1_fft = np.fft.fftshift(np.fft.fft(bias1, axis=-1))
                bn2_fft = np.fft.fftshift(np.fft.fft(bias2, axis=-1))

                plt.figure(figsize=(8,8))
                plt.subplot(2,2,1)
                plt.plot(np.abs(bias1)[0], label='X')
                plt.plot(np.abs(bias1)[1], label='Y')
                plt.title('BN1 Bias')
                plt.legend()
                plt.subplot(2,2,2)
                plt.plot(np.abs(bias2)[0], label='X')
                plt.plot(np.abs(bias2)[1], label='Y')
                plt.title('BN2 Bias')
                plt.legend()
                plt.subplot(2,2,3)
                plt.plot(np.abs(bn1_fft)[0], label='FFT_X')
                plt.plot(np.abs(bn1_fft)[1], label='FFT_Y')
                plt.title('BN1 Bias fft')
                plt.legend()
                plt.subplot(2,2,4)
                plt.plot(np.abs(bn2_fft)[0], label='FFT_X')
                plt.plot(np.abs(bn2_fft)[1], label='FFT_Y')
                plt.title('BN2 Bias fft')
                plt.legend()
                plt.savefig(save_path+'bn_bias.png', dpi = 500)
            else:
                bn1_var = np.array(bn[0])
                bn2_var = np.array(bn[1])
            plt.figure(figsize=(8,4))
            plt.subplot(1,2,1)
            plt.plot(bn1_var[:,0].real, label='var1')
            plt.plot(bn1_var[:,1].real, label='var2')
            plt.title('BN1 Var')
            plt.subplot(1,2,2)
            plt.plot(bn2_var[:,0].real, label='var1')
            plt.plot(bn2_var[:,1].real, label='var2')
            plt.title('BN2 Var')
            plt.legend()
            plt.savefig(save_path+'bn_var.png', dpi = 500)
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.plot(mimo_td_window[0])
        plt.title('X pol.')
        plt.subplot(1,2,2)
        plt.plot(mimo_td_window[1])
        plt.title('Y pol.')
        plt.savefig(save_path+'mimo_td_window.png', dpi = 500)
        s1, s2 = mimo_fd.shape[0], mimo_fd.shape[1]
        if s1 == 2 and s2 == 2:
            data_lable = ['XX', 'YX',\
                          'XY', 'YY']
        elif s1 == 4 and s2 == 2:
            data_lable = ['XXr', 'YXr',\
                          'XXi', 'YXi',\
                          'XYr', 'YYr',\
                          'XYi', 'YYi']   
        elif s1 == 4 and s2 == 4:
            data_lable = ['XrXr', 'XiXr', 'YrXr', 'YiXr',\
                          'XrXi', 'XiXi', 'YrXi', 'YiXi',\
                          'XrYr', 'XiYr', 'YrYr', 'YiYr',\
                          'XrYi', 'XiYi', 'YrYi', 'YiYi']
        else:
            raise ValueError('mimo_fd shape error')
        # mimo_fd plot
        plt.figure(figsize=(4*s2,4*s1))
        for i in range(s1):
            for j in range(s2):
                plot_label = data_lable[i*s2+j]
                ax = plt.subplot(s1,s2,i*s2+j+1)
                ax.plot(mimo_fd_abs[i,j], label='Amplitude')
                ax.legend(loc='center left')
                ax_r = ax.twinx()
                ax_r.plot(mimo_fd_phase[i,j], '--', color='red', label='Phase')
                ax_r.legend(loc='center right')
                plt.title(plot_label)
        plt.tight_layout()
        plt.savefig(save_path+'mimo_fd_taps.png', dpi=500)
        # plot mimo_td
        plt.figure(figsize=(4*s2,4*s1))
        for i in range(s1):
            for j in range(s2):
                plot_label = data_lable[i*s2+j]
                ax = plt.subplot(s1,s2,i*s2+j+1)
                ax.plot(mimo_td[i,j].real, label='Real_'+plot_label)
                ax.plot(mimo_td[i,j].imag, label='Imag_'+plot_label)
                plt.legend()
        plt.tight_layout()
        plt.savefig(save_path+'mimo_td_taps.png', dpi=500)
        # plot cdc_taps
        plt.figure(figsize=(10,10))
        plt.subplot(2,2,1)
        plt.plot(np.abs(cdc_tap[0]))
        plt.title('X Amplitude')
        plt.subplot(2,2,2)
        plt.plot(cdc_tap_phase[0])
        plt.title('X Phase')
        plt.subplot(2,2,3)
        plt.plot(np.abs(cdc_tap[1]))
        plt.title('Y Amplitude')
        plt.subplot(2,2,4)
        plt.plot(cdc_tap_phase[1])
        plt.title('Y Phase')
        plt.savefig(save_path+'cdc_taps.png', dpi=500)

        plt.figure(figsize=(4,4))
        plt.plot(lofo)
        plt.ylabel('GHz')
        plt.xlabel('Iterations')
        plt.savefig(save_path+'clofo.png', dpi=500)
    
        plt.figure(figsize=(12,8))
        plt.subplot(2,3,1)
        plt.plot(skew[0])
        plt.title('XI Skew')
        plt.subplot(2,3,2)
        plt.plot(skew[1])
        plt.title('XQ Skew')
        plt.subplot(2,3,3)
        plt.plot(skew[1]-skew[0])
        plt.title('XQ-XI Skew')
        plt.subplot(2,3,4)
        plt.plot(skew[2])
        plt.title('XI Skew')
        plt.subplot(2,3,5)
        plt.plot(skew[3])
        plt.title('YQ Skew')
        plt.subplot(2,3,6)
        plt.plot(skew[3]-skew[2])
        plt.title('YQ-YI Skew')
        plt.savefig(save_path+'skew.png', dpi=500)
        if mimo_skew is not None:
            plt.figure(figsize=(12,8))
            plt.subplot(2,3,1)
            plt.plot(mimo_skew[0])
            plt.title('XI Skew')
            plt.subplot(2,3,2)
            plt.plot(mimo_skew[1])
            plt.title('XQ Skew')
            plt.subplot(2,3,3)
            plt.plot(mimo_skew[1]-mimo_skew[0])
            plt.title('XQ-XI Skew')
            plt.subplot(2,3,4)
            plt.plot(mimo_skew[2])
            plt.title('XI Skew')
            plt.subplot(2,3,5)
            plt.plot(mimo_skew[3])
            plt.title('YQ Skew')
            plt.subplot(2,3,6)
            plt.plot(mimo_skew[3]-mimo_skew[2])
            plt.title('YQ-YI Skew')
            plt.savefig(save_path+'mimo_skew.png', dpi=500)

        if frame is not None:
            frame_plot(frame, save_path)
        plt.close('all')

        #style:字体的类型，有normal,oblique(斜体),italic(斜体)；
#weight:字体的粗细，有'normal','light','medium','semibold','bold','heavy','black'；
#204
# fontscale = 1.8
# plt.figure(figsize = (18, 9))
#         plt.subplot(1,2,1)
#         plt.grid()
#         ax = sns.histplot(x = y.real, bins = 1000) # , color = 'orange'
#         # ax = sns.histplot(x = y.real, bins = 1000, kde = True, kde_kws = {bd_adjust:0.2})
#         plt.xticks(rotation = tick_rot)
#         plt.xlabel('In-phase',fontsize = 'large')
#         plt.ylabel('Symbol error number',fontsize = 'large')
#         plt.xticks(rotation = tick_rot,fontsize='18')
#         plt.yticks(rotation = tick_rot,fontsize='18')
#         plt.subplot(1,2,2)
#         plt.grid()
#         plt.xticks(rotation = tick_rot,fontsize='18')
#         plt.yticks(rotation = tick_rot,fontsize='18')
#         ax = sns.histplot(x = y.imag, bins = 1000)
#         plt.xlabel('Quadrature',fontsize = 'large')
#         plt.ylabel('Symbol error number' ,fontsize = 'large')
#         plt.savefig(name+'_rxsym.png', dpi = 500)