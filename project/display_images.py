# -*- coding: utf-8 -*-  # Λ
'''
display jet images

use a python console (not ipython)

green box means isPU==True
red box means isPU==False

controls
--------
go forward:     enter
go back:        backspace
close figure:   esc
seek:           type a number and enter
'''

import numpy as np
import matplotlib.pyplot as pl

np.set_printoptions(precision=3)

def plot_cdf_of_max(snippets):  # 10640759
    NN = len(snippets)
    mx = np.empty(NN)
    for idx in range(NN):
        mx[idx] = np.max(snippets[idx])
    # pl.hist(mx, 20)
    mx = np.sort(mx)
    yy = np.array(range(NN))/float(NN)
    pl.plot(mx, yy)
    pl.show()

def show_fig(axs, ncols, nrows, x, y, idstart, vmax):
    ids = np.arange(idstart, idstart+ncols*nrows) % x.shape[0]
    for idy in range(nrows):
        for idx in range(ncols):
            idxy = idx+idy*ncols
            axt = axs[idxy]
            axt.cla() # need this to display correct pixel values at mouse over
            axt.imshow(
                    np.transpose(x[ids[idxy]], (1, 2, 0)),
                    interpolation='none', cmap='gray', aspect=1,
                    vmin=0, vmax=vmax)  # cmap='viridis'
            axt.tick_params(axis='both', which='both',
                    bottom='off', top='off', left='off', right='off',
                    labelbottom='off', labelleft='off')
            axt.get_yaxis().set_visible(False)
            axt.get_xaxis().set_visible(False)

            if y[ids[idxy]]:
                [sp.set_edgecolor('green') for sp in axt.spines.values()]
                [sp.set_linewidth(3) for sp in axt.spines.values()]
            else:
                [sp.set_edgecolor('red') for sp in axt.spines.values()]
                [sp.set_linewidth(3) for sp in axt.spines.values()]
                # 7778954 1639463
    axs[0].set_title(str(ids[0]))
    pl.pause(.01)  # 11874767 22873410
    # https://github.com/matplotlib/matplotlib/issues/1646/
    # to control ticks. 12998430
    # fig1.colorbar(obj1)
    # obj3.set_clim([dataiframe.min(), dataiframe.max()])  # 21912859

def on_key_release(event, fig, num, axs, ncols, nrows, x, y, idstart, vmax):
    # idstart: np array
    # idstart[0]: counter
    # num: list
    # num[0]: string
    ky = event.key
    if ky == 'enter':
        if len(num[0]):  # seek
            idstart[0] = int(num[0]) % x.shape[0]
            num[0] = ''
        else:
            idstart[0] += ncols*nrows
    elif ky == 'backspace':  # back
        idstart[0] -= ncols*nrows
    elif ky == 'escape':  # quit
        pl.close(fig)
    else:
        if len(ky) == 1:  # seek
            if (ord('0') <= ord(ky) and ord(ky) <= ord('9')):
                num[0] += ky
    idstart[0] %= x.shape[0]
    show_fig(axs, ncols, nrows, x, y, idstart, vmax)
    # print('you released', event.key, event.xdata, event.ydata)

def eventcontrol(x, y):
    ncols = 6
    nrows = 4
    vmax = 10#23#np.max(x)

    fig = pl.figure(figsize=(ncols, nrows))
    axs = []
    for idy in range(nrows):
        for idx in range(ncols):
            ax1 = pl.subplot2grid((nrows, ncols), (idy, idx))
            axs.append(ax1)
    idstart = np.zeros(1, dtype='int')
    num = ['']

    show_fig(axs, ncols, nrows, x, y, idstart, vmax)
    # http://matplotlib.org/users/event_handling.html
    # 24960910
    fig.canvas.mpl_connect(
            'key_release_event',
            lambda event: on_key_release(
                                event, fig, num,
                                axs, ncols, nrows, x, y, idstart, vmax))

if __name__ == '__main__':
    data_loc = 'data/'
    xfile = 'pixel_image_clus_trks_j0_EM_train.npy'
    yfile = 'isPU_j0_EM_train.npy'
    
    x = np.load(data_loc+xfile)
    y = np.load(data_loc+yfile)
    #raise Exception('')
    
    #plot_cdf_of_max(x)
    
    eventcontrol(x, y)
