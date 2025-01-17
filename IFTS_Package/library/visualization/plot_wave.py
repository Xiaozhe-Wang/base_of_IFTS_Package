import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def wave1d_plot(x, axs, title, smooth = 1, EMA_SPAN = 15, *args, **kwargs):
    
    if smooth:
        colors = sns.color_palette()
        axs.plot(x, alpha=0.4, color=colors[0])[0]
        x_smooth = pd.Series(x).ewm(span=EMA_SPAN).mean()
        axs.plot(x_smooth, color=colors[0])
    else:
        axs.plot(x)
    if title is not None:
        axs.set_title(title, fontsize = 'x-large')
    
    return axs
        





if __name__ == '__main__':
    t = np.arange(0,20,0.1)
    x = np.exp(-t)
    n = np.random.randn(x.shape[0])
    y = x + 0.1 * n
    fig, axs = plt.subplots()
    axs = wave1d_plot(y, axs, 'noisy exp')
    plt.show()