import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_psd(x, fs, axs, Fc = 0):
    
    axs.psd(x, 4096, Fs = fs, Fc = Fc)
    axs.grid(True)
    axs.set_ylabel('')
    axs.set_xlabel('Frequency(THz)')
    return axs