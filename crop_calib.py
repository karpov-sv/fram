#!/usr/bin/env python

import numpy as np
import os, sys

from astropy.io import fits

import warnings
from astropy.wcs import FITSFixedWarning

warnings.simplefilter('ignore', FITSFixedWarning)


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog [options] arg")
    parser.add_option('--x0', help='X0', action='store', dest='x0', type='int', default=0)
    parser.add_option('--y0', help='Y0', action='store', dest='y0', type='int', default=0)
    parser.add_option('--width', help='Width', action='store', dest='width', type='int', default=1000)
    parser.add_option('--height', help='Height', action='store', dest='height', type='int', default=1000)
    parser.add_option('-r', '--replace', help='Whether to overwrite existing file', action='store_true', dest='replace', default=False)

    (options,args) = parser.parse_args()

    for filename in args:
        outname = os.path.split(filename)[1]
        s = outname.split('_')
        s[-2] = f"{options.width}x{options.height}"
        outname = '_'.join(s)
        outname = os.path.join(
            os.path.split(filename)[0],
            outname
        )
        print(filename, '->', outname)

        if not options.replace and os.path.exists(outname):
            continue

        image,header = fits.getdata(filename, -1, header=True)

        image1 = image[options.y0:options.y0+options.height, options.x0:options.x0+options.width]

        print(image.shape[1], 'x', image.shape[1], '->', image1.shape[1], image1.shape[0], '+', options.x0, options.y0)

        fits.writeto(outname, image1, header, overwrite=options.replace)
