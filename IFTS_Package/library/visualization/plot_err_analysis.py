from math import log10
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def bn(x, sig_p):
    p = np.mean(np.abs(x) ** 2)
    return x / (p / sig_p) ** 0.5

def density_esti(x, y):
    x_map = np.unique(x)
    map_class_num = x_map.shape[0]
    density = np.zeros(map_class_num)
    i = 0
    for x_ in x_map:
        num = np.sum(np.where(x == x_, 1, 0))
        density[i] = np.sum(np.where(y == x_, 1, 0)) / num
        i += 1
    return density, x_map

def circle_cumulation(r, err_rate, sym_map):
    
    pass

def err_analysis_plot(y, sym_error, sym_map, name = './hist'):
    # font = {'family' : 'Times New Roman',
    font = {
        'weight' : 'normal', # bold 加粗, normal(default) light, ultralight (0-1000)
        'size'   : 15}
    # plt.rc('font', **font) 
    
    class_num = sym_map.shape[0]
    real = (sym_map.real) * (class_num) ** 0.5 
    imag = (sym_map.imag) * (class_num) ** 0.5 
    amplitude = ((real ** 2 + imag ** 2) * 100).astype(np.int64)
    sort_idx  = np.argsort(amplitude)
    # 
    real_error = (real[sym_error] * 100).astype(np.int64) / 100
    imag_error = (imag[sym_error] * 100).astype(np.int64) / 100
    amplitude_error = (amplitude[sym_error] * 100).astype(np.int64) / 100
    # Construct a data frame of the symbol errors
    # data_err_sym = pd.Series(sym_error).value_counts()
    # print(data_err_sym)
    idx = np.arange(class_num)
    data_err_num = np.zeros(class_num)
    # amplitude_err_num = np.zeros(amplitude_err_num.shape[0])
    for i in range(class_num):
        data_err_num[i] = np.sum(np.where((sym_error == i), 1, 0))
        # amplitude_err_num[i] = np.sum(np.where((sym_error == i), 1, 0))
    # df = pd.DataFrame({'Constellation index':idx, 'Sym Err Num':data_err_num,\
    #     'real': (sym_map.real * int(np.log2(class_num))).astype(np.int64), \
    #         'imag': (sym_map.imag* int(np.log2(class_num))).astype(np.int64), \
    #             'abs': (np.abs(sym_map)* int(np.log2(class_num))).astype(np.int64) })
    # r = df['real'].value_counts()
    tick_rot = - 30
    sns.set_theme(style="ticks", font_scale=1.8)
    plt.rcParams['font.family'] = 'Times New Roman'#修改字体类型为times new roman
    plt.figure(figsize = (18, 16))
    
    plt.subplot(2,2,1)
    plt.grid()
    ax = sns.countplot(x = real_error)
    plt.xticks(rotation = tick_rot)
    plt.xlabel('In-phase')
    plt.ylabel('Symbol error number')
    plt.subplot(2,2,2)
    plt.grid()
    plt.xticks(rotation = tick_rot)
    ax = sns.countplot(x = imag_error)
    plt.xlabel('Quadrature')
    plt.ylabel('Symbol error number')
    plt.subplot(2,2,3)
    plt.grid()
    ax = sns.barplot(x = idx, y = data_err_num)
    # plt.xticks(rotation = tick_rot)
    plt.xticks([])
    plt.xlabel('Label index (0-'+str(class_num-1)+')')
    plt.ylabel('Symbol error number')
    plt.subplot(2,2,4)
    plt.grid()
    ax = sns.barplot(x = amplitude[sort_idx], y = data_err_num[sort_idx])       # calculate the mean error of each amplitude
    idx_range = np.arange(np.unique(amplitude).shape[0])
    plt.xticks(idx_range, idx_range + 1, rotation = tick_rot)
    plt.xlabel('Relative Amplitude')
    plt.ylabel('Symbol error number per amplitude')
    plt.savefig(name+'.png', dpi = 500)

    plt.figure(figsize = (18, 9))
    plt.subplot(1,2,1)
    plt.grid()
    ax = sns.histplot(x = y.real, bins = 1000) # , color = 'orange'
        # ax = sns.histplot(x = y.real, bins = 1000, kde = True, kde_kws = {bd_adjust:0.2})
    plt.xticks(rotation = tick_rot)
    plt.xlabel('In-phase',fontsize = 'large')
    plt.ylabel('Symbol error number',fontsize = 'large')
    plt.xticks(rotation = tick_rot,fontsize='18')
    plt.yticks(rotation = tick_rot,fontsize='18')
    plt.subplot(1,2,2)
    plt.grid()
    plt.xticks(rotation = tick_rot,fontsize='18')
    plt.yticks(rotation = tick_rot,fontsize='18')
    ax = sns.histplot(x = y.imag, bins = 1000)
    plt.xlabel('Quadrature',fontsize = 'large')
    plt.ylabel('Symbol error number' ,fontsize = 'large')
    plt.savefig(name+'_rxsym.png', dpi = 500)

    plt.figure(figsize = (9, 9))
    # plt.grid()
    # sns.scatterplot(x = real, y = imag, hue = data_err_num, \
        # size = data_err_num, sizes=(200, 400), palette = 'plasma_r')    # color+_r
    # plt.scatter(real, imag,  c = 'r', marker = '+')
    # sns.set_theme(style="ticks")
    sns.set_theme(style="darkgrid")
    sns.scatterplot(x = real, y = imag, hue = data_err_num, size = data_err_num, \
        sizes=(100, 300), legend = False)     #  palette = 'plasma_r'
    # plt.grid()
    plt.savefig(name+'_constell.png', dpi = 500)



def err_analysis_plot_v2(rx, x, y, sym_error, sym_map, hist = 1, name = './hist'):
    """
    Input:
        rx: rx symbol
        x: tx symbol index
        y: symbol error
        sym_error: symbol erro index
    """
    # font = {'family' : 'Times New Roman',
    font = {
        'weight' : 'normal', # bold 加粗, normal(default) light, ultralight (0-1000)
        'size'   : 15}
    # plt.rc('font', **font) 
    tx_sym = sym_map[x]
    class_num = sym_map.shape[0]
    real_error, real_map = density_esti(tx_sym.real, sym_map.real[sym_error])
    imag_error, imag_map = density_esti(tx_sym.imag, sym_map.imag[sym_error])
    real_map = (real_map * 100).astype(np.int64) / 100
    imag_map = (imag_map * 100).astype(np.int64) / 100
    amplitude = (sym_map.real ** 2 + sym_map.imag ** 2)
    amplitude_error = (amplitude[sym_error] * 100).astype(np.int64) / 100
    sort_idx  = np.argsort(amplitude)
    idx = np.arange(class_num)
    data_err_num = np.zeros(class_num)
    data_num = np.zeros(class_num) + 1e-8
    # amplitude_err_num = np.zeros(amplitude_err_num.shape[0])
    r"Find the data error rate according to the amplitude with nature index"
    amplitude_err_sort = np.zeros_like(sym_error)
    snr_lin = np.zeros(class_num)
    for i in range(class_num):
        amplitude_err_sort[sym_error == i] = sort_idx[i]
        data_err_num[i] = np.sum(np.where((sym_error == i), 1, 0))
        data_num[i] = np.sum(np.where((x == i), 1, 0))
        snr_lin[i] = np.mean(np.abs(rx[x == i] - sym_map[i]) ** 2)
        # amplitude_err_num[i] = np.sum(np.where((sym_error == i), 1, 0))
    snr = -10 * np.log10(snr_lin)
    data_err_rate = data_err_num / data_num
    """
    Figure plot 
    """
    if hist:
        color = sns.color_palette("Set2")
        tick_rot = - 60
        sns.set_theme(style="ticks", font_scale=1.25)
        plt.figure(figsize = (18, 16))
        # In phase error histgram
        plt.subplot(2,2,1)
        plt.grid()
        ax = sns.barplot(x = real_map, y = real_error, palette=color)
        plt.xticks(rotation = tick_rot) 
        plt.xlabel('In-phase')
        plt.ylabel('Symbol error rate')

        # Quadrature error histgram
        plt.subplot(2,2,2)
        plt.grid()
        plt.xticks(rotation = tick_rot)
        ax = sns.barplot(x = imag_map, y = imag_error, palette=color)
        plt.xlabel('Quadrature')
        plt.ylabel('Symbol error rate')

        # Relative amplitude error histgram
        plt.subplot(2,2,3)
        plt.grid()
        ax = sns.barplot(x = amplitude[sort_idx], y = data_err_rate[sort_idx], palette=color)       # calculate the mean error of each amplitude
        idx_range = np.arange(np.unique(amplitude).shape[0])
        plt.xticks(idx_range, idx_range + 1, rotation = tick_rot)
        plt.xlabel('Relative Amplitude')
        plt.ylabel('Symbol error rate per amplitude')

        # Amplitude error histgram
        plt.subplot(2,2,4)
        plt.grid()
        ax = sns.barplot(x = np.arange(class_num), y = data_err_rate[sort_idx], palette = 'rocket_r')
        plt.xticks([])
        ax.set_ylabel('Symbol error rate')
        ax2 = ax.twinx()
        sns.ecdfplot(x = amplitude_err_sort, ax = ax2)
        ax2.set_ylabel('')
        ax.set_xlabel('Relative Amplitude\nIndex (0-'+str(class_num-1)+')')
        plt.xticks([])

        plt.savefig(name+'.png', dpi = 500)
        

        
        plt.figure(figsize = (18, 9))
        plt.subplot(1,2,1)
        plt.grid()
        ax = sns.histplot(x = y.real, bins = 1000) # , color = 'orange'
        # ax = sns.histplot(x = y.real, bins = 1000, kde = True, kde_kws = {bd_adjust:0.2})
        plt.xticks(rotation = tick_rot)
        plt.xlabel('In-phase')
        plt.ylabel('Symbol error number')
        plt.subplot(1,2,2)
        plt.grid()
        plt.xticks(rotation = tick_rot)
        ax = sns.histplot(x = y.imag, bins = 1000)
        plt.xlabel('Quadrature')
        plt.ylabel('Symbol error number')
        plt.savefig(name+'_rxsym.png', dpi = 500)


        # sns.set_theme(style="ticks")
        sns.set_theme(style="darkgrid")
        # sns.scatterplot(x = sym_map.real, y = sym_map.imag, hue = data_err_num, size = data_err_num, \
        #     sizes=(100, 300), legend = False)     #  palette = 'plasma_r'
        sns.relplot(x = sym_map.real, y = sym_map.imag, hue = data_err_rate, size = data_err_rate, \
            sizes=(100, 200), kind = 'scatter')     #  palette = 'plasma_r'
        # plt.title('SER of each constellation points')
        plt.savefig(name+'_constell_ser.png', dpi = 500)

    sns.set_theme(style="darkgrid")
    g = sns.relplot(x = sym_map.real, y = sym_map.imag, hue = snr, size = snr, \
        sizes=(100, 200), kind = 'scatter', palette='crest_r')     #  palette = 'plasma_r'
    (g.set_titles("SNR of each constellation points")
    .tight_layout(w_pad=0))
    plt.savefig(name+'_constell_snr.png', dpi = 500)

if __name__=='__main__':
    penguins = sns.load_dataset("penguins")
    sns.ecdfplot(data=penguins, x="flipper_length_mm")