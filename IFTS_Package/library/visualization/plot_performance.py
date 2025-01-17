import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def ber_plot(x, y, para, name = 'BER'):
    fig, axs = plt.subplots(figsize=(8, 6))
    axs.semilogy(x, y)
    axs.scatter(x, y)
    axs.grid()
    axs.set_xlabel('Input Power (dBm)', fontsize = 'large', fontstyle = 'italic')
    axs.set_ylabel('BER', fontsize = 'large', fontstyle = 'italic')
    axs.set_title(name, fontsize = 'x-large')
    axs.tick_params(labelsize = 'large')
    fig.savefig(para.path + name + '.png', dpi = 500)
    plt.close(fig)

def qfactor_plot(x, y, para, name = 'qfactor', ylabel = 'Q factor'):
    fig, axs = plt.subplots(figsize=(8, 6))
    axs = qfactor_plot_axs(x, y, axs)
    axs.grid()
    axs.set_xlabel('Input Power (dBm)', fontsize = 'large', fontstyle = 'italic')
    axs.set_ylabel(ylabel, fontsize = 'large', fontstyle = 'italic')
    axs.set_title(name, fontsize = 'x-large')
    axs.tick_params(labelsize = 'large')
    fig.savefig(para.path + name + '.png', dpi = 500)
    plt.close(fig)

def qfactor_plot_axs(x, y, axs):
    axs.plot(x, y)
    axs.scatter(x, y)
    return axs


