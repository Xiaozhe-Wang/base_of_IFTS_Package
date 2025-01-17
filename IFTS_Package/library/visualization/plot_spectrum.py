
def psd():
    plt.figure(figsize = (5, sig.nPol * 5.2))
    p = ['x', 'y']
    for i_p in range(sig.nPol):
        plt.subplot(2, 1, i_p + 1)
        temp = sig.rx_sig[i_p].cpu().numpy()[:, 0] + 1j * sig.rx_sig[i_p].cpu().numpy()[:, 1]
        plt.psd(temp, 4196, Fs = sig.sam_rate)
        plt.title('PSD '+p[i_p])
        xlim = sig.channel_space * sig.channel_num / 2 + 50
        plt.xlim(-xlim, xlim)
    plt.suptitle('PSD')
    plt.savefig(sig.figpath + 'power_spectrum_density.png')
    plt.close()