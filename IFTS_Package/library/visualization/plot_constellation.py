import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator

def scatter_plot_nPol(x, para, sam_num = 1, name = 'constellation'):
    nPol = len(x)
    pol_name = ["X Polarization", "Y Polarization"]
    fig, axs = plt.subplots(1, nPol, figsize=(6* nPol, 6))
    fig.suptitle(name)
    # fig.set_facecolor('#282828')
    idx = para.get_idx(sam_num = 1)
    
    for i_p in range(nPol):
        y = x[i_p][idx]
        if hasattr(para, 'colour_seq'):
            c = para.colour_seq[para.select_wave][i_p]
        else:
            c = None
        axs[i_p] = scatter_plot(y, axs[i_p], pol_name[i_p], c=c)
    fig.savefig(para.path + name + '.png', facecolor = fig.get_facecolor(), transparent=True, dpi = 500)
    plt.close(fig)

def scatter_plot(x, axs, title, c=None, xlim=None, ylim=None, s=None):
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    xy_max = max(np.max(np.abs(x.real))\
            , np.max(np.abs(x.imag)))
    if xlim == None:
        xlim = [-xy_max, xy_max]
    if ylim == None:
        ylim = [-xy_max, xy_max]
    if s == None:
        s = 10
    axs.scatter(x.real, x.imag, c = c, s = s)
    # axs.set_xlim(xlim[0], xlim[1])
    # axs.set_ylim(ylim[0], ylim[1])
    # axs.grid(linestyle='--')
    # axs.set_xlabel('In-Phase', fontsize = 'large', fontstyle = 'italic')
    # axs.set_ylabel('Quadrature', fontsize = 'large', fontstyle = 'italic')
    # axs.set_title(title, fontsize = 'x-large')
    # axs.tick_params(labelsize = 'large')

    return axs

def scatter_plot_white(x, axs, title, c=None, xlim=None, ylim=None, s=None):
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    xy_max = max(np.max(np.abs(x.real))\
            , np.max(np.abs(x.imag)))
    if xlim == None:
        xlim = [-xy_max, xy_max]
    if ylim == None:
        ylim = [-xy_max, xy_max]
    if s == None:
        s = 10
    # axs.set_facecolor('k')
    axs.scatter(x.real, x.imag, c = c, s = s)
    axs.set_xlim(xlim[0], xlim[1])
    axs.set_ylim(ylim[0], ylim[1])
    axs.grid(linestyle='--')
    label_c = '#000000'
    axs.set_xlabel('In-Phase', color = label_c, fontsize = 'large', fontstyle = 'italic')
    axs.set_ylabel('Quadrature', color = label_c, fontsize = 'large', fontstyle = 'italic')
    axs.set_title(title, color = label_c, fontsize = 'x-large')
    # axs.spines['left'].set_color(label_c)
    # axs.spines['right'].set_color(label_c)
    # axs.spines['top'].set_color(label_c)
    # axs.spines['bottom'].set_color(label_c)
    # axs.tick_params(labelcolor= label_c, labelsize = 'large')
    return axs

def scatter_plot_black(x, axs, title, c=None, xlim=None, ylim=None, s=None):
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    xy_max = max(np.max(np.abs(x.real))\
            , np.max(np.abs(x.imag)))
    if xlim == None:
        xlim = [-xy_max, xy_max]
    if ylim == None:
        ylim = [-xy_max, xy_max]
    if s == None:
        s = 10
    axs.set_facecolor('k')
    axs.scatter(x.real, x.imag, c = c, s = s)
    axs.set_xlim(xlim[0], xlim[1])
    axs.set_ylim(ylim[0], ylim[1])
    axs.grid(linestyle='--')
    label_c = '#B4B6BD'
    axs.set_xlabel('In-Phase', color = label_c, fontsize = 'large', fontstyle = 'italic')
    axs.set_ylabel('Quadrature', color = label_c, fontsize = 'large', fontstyle = 'italic')
    axs.set_title(title, color = label_c, fontsize = 'x-large')
    axs.spines['left'].set_color(label_c)
    axs.spines['right'].set_color(label_c)
    axs.spines['top'].set_color(label_c)
    axs.spines['bottom'].set_color(label_c)
    axs.tick_params(labelcolor= label_c, labelsize = 'large')
    return axs

def scatter_plot_black(x, axs, title, c=None, xlim=None, ylim=None, s=None):
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    xy_max = max(np.max(np.abs(x.real))\
            , np.max(np.abs(x.imag)))
    if xlim == None:
        xlim = [-xy_max, xy_max]
    if ylim == None:
        ylim = [-xy_max, xy_max]
    if s == None:
        s = 10
    axs.set_facecolor('k') 
    axs.scatter(x.real, x.imag, c = c, s = s)
    axs.set_xlim(xlim[0], xlim[1])
    axs.set_ylim(ylim[0], ylim[1])
    axs.grid(linestyle='--')
    label_c = '#B4B6BD'
    axs.set_xlabel('In-Phase', color = label_c, fontsize = 'large', fontstyle = 'italic')
    axs.set_ylabel('Quadrature', color = label_c, fontsize = 'large', fontstyle = 'italic')
    axs.set_title(title, color = label_c, fontsize = 'x-large')
    axs.spines['left'].set_color(label_c)
    axs.spines['right'].set_color(label_c)
    axs.spines['top'].set_color(label_c)
    axs.spines['bottom'].set_color(label_c)
    axs.tick_params(labelcolor= label_c, labelsize = 'large')
    return axs

def scatter_heat_map(x, axs, title,  xlim=None, ylim=None, s=None):
    xy_max = max(np.max(np.abs(x.real))\
            , np.max(np.abs(x.imag)))
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if xlim == None:
        xlim = [-xy_max, xy_max]
    if ylim == None:
        ylim = [-xy_max, xy_max]
    if s == None:
        s = 10
    cmap = sns.cubehelix_palette(as_cmap = True, dark = 0, light = 1, reverse=True)
    # cmap = 'autumn_r'
    # axs.scatter(x.real, x.imag, c = 'b', s = s)
    axs = sns.kdeplot(x = x.real, y = x.imag, ax = axs, levels = 50, thresh = 0.14, fill=True, cmap = cmap, alpha = 1)
    axs.set_facecolor('k')
    axs.set_xlim(xlim[0], xlim[1])
    axs.set_ylim(ylim[0], ylim[1])
    axs.grid(linestyle='--')
    label_c = '#B4B6BD'
    axs.set_xlabel('In-Phase', color = label_c, fontsize = 'large', fontstyle = 'italic')
    axs.set_ylabel('Quadrature', color = label_c, fontsize = 'large', fontstyle = 'italic')
    axs.set_title(title, color = label_c, fontsize = 'x-large')
    axs.spines['left'].set_color(label_c)
    axs.spines['right'].set_color(label_c)
    axs.spines['top'].set_color(label_c)
    axs.spines['bottom'].set_color(label_c)
    axs.tick_params(labelcolor= label_c, labelsize = 'large')


def scatter_plot_white_density(x, axs, title, searchR=None, c=None, xlim=None, ylim=None, s=None):
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    xy_max = max(np.max(np.abs(x.real))\
            , np.max(np.abs(x.imag)))
    scal = 0.01
    xlim = [-xy_max-scal, xy_max+scal]
    ylim = [-xy_max-scal, xy_max+scal]
    
    if s == None:
        s = 1
    if searchR==None: 
        searchR=0.1
    #------calculate desity--------------
    Dreal = x.real
    Dimag = x.imag
    xyz = np.zeros_like(Dreal)
    
    tmpList = np.arange(int(Dreal.size))
    index_i = []
    for i in range(0,int(Dreal.size)):
        index_i = tmpList[(Dreal>Dreal[i]-searchR) & (Dreal<Dreal[i]+searchR) & (Dimag> Dimag[i]-searchR) & (Dimag < Dimag[i]+searchR)]
        sizeIndexI = len(index_i)
        xyz[i] = sizeIndexI

    idx = xyz.argsort() #[500,] 
    Dreal, Dimag, xyz = Dreal[idx], Dimag[idx], xyz[idx] # 从小到大排列
    #--------------scatter plot----------------
    axs.scatter(Dreal, Dimag, c = xyz, s = s,cmap="mako")# cmap="mako"
    axs.set_xlim(xlim[0], xlim[1])
    axs.set_ylim(ylim[0], ylim[1])
    a = (2*xy_max+2*scal)/4
    xmajorLocator = MultipleLocator(a)  
    axs.xaxis.set_major_locator(xmajorLocator)
    # xmajorFormatter = FormatStrFormatter('%0.0f')  # 设置x轴标签文本的格式
    # axs.xaxis.set_major_formatter(xmajorFormatter)
    ymajorLocator = MultipleLocator(a)  # 将x主刻度标签间隔数
    axs.yaxis.set_major_locator(ymajorLocator)
    # ymajorFormatter = FormatStrFormatter('%0.0f')  # 设置x轴标签文本的格式
    # axs.yaxis.set_major_formatter(ymajorFormatter)
    axs.set_xticklabels([])#只关闭label
    axs.set_yticklabels([])
    # axs.set_xticks([])
    # axs.set_yticks([])

    axs.grid(linestyle='--')
    # a = np.linspace(-xy_max,xy_max,5)
    # axs.set_xticks(a,())
    # axs.set_yticks(a,())
    # ticks_label = axs.get_xticklabels() + axs.get_yticklabels()
    # [ticks_label_temp.set_font(font3) for ticks_label_temp in ticks_label]
    
    return axs