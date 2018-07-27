#!/usr/bin/env/python

import numpy as np
import matplotlib.pyplot as plt
plt.rc('image', interpolation='nearest', origin='lower', cmap = 'hot')
rcParams = plt.rcParams.copy()

from mpl_toolkits.axes_grid1 import make_axes_locatable

def colorbar(obj=None, ax=None, size="5%", pad=0.1):
    should_restore = False

    if obj is not None:
        ax = obj.axes
    elif ax is None:
        ax = plt.gca()
        should_restore = True

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)

    plt.colorbar(obj, cax=cax)

    if should_restore:
        plt.sca(ax)

def imshow(image, qq=[0.5,97.5], show_colorbar=True, **kwargs):
    vmin,vmax = np.percentile(image[np.isfinite(image)], qq)
    plt.imshow(image, vmin=vmin, vmax=vmax, **kwargs)
    if show_colorbar:
        colorbar()

def breakpoint():
    try:
        from IPython.core.debugger import Tracer
        Tracer()()
    except:
        import pdb
        pdb.set_trace()
