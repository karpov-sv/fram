#!/usr/bin/env/python

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import binned_statistic_2d

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
    vmin1,vmax1 = np.percentile(image[np.isfinite(image)], qq)
    if not kwargs.has_key('vmin'):
        kwargs['vmin'] = vmin1
    if not kwargs.has_key('vmax'):
        kwargs['vmax'] = vmax1
    plt.imshow(image, **kwargs)
    if show_colorbar:
        colorbar()

def breakpoint():
    try:
        from IPython.core.debugger import Tracer
        Tracer()()
    except:
        import pdb
        pdb.set_trace()

def binned_map(x, y, value, bins=16, statistic='mean', qq=[0.5, 97.5], show_colorbar=True, show_dots=False):
    gmag0, xe, ye, binnumbers = binned_statistic_2d(x, y, value, bins=bins, statistic=statistic)

    limits = np.percentile(gmag0[np.isfinite(gmag0)], qq)

    plt.imshow(gmag0.T, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]], interpolation='nearest', vmin=limits[0], vmax=limits[1], aspect='auto')
    if show_colorbar:
        plt.colorbar()

    if show_dots:
        plt.autoscale(False)
        plt.plot(x, y, 'b.', alpha=0.3)
