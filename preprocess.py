#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, glob

from astropy.wcs import WCS
from astropy.io import fits

import numpy as np

from fram.fram import Fram
from fram import calibrate

if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog [options] from to")
    parser.add_option('-d', '--db', help='Database name', action='store', dest='db', type='str', default='fram')
    parser.add_option('-H', '--host', help='Database host', action='store', dest='dbhost', type='str', default=None)
    parser.add_option('-v', '--verbose', help='Verbose', action='store_true', dest='verbose', default=False)

    (options,files) = parser.parse_args()

    if len(files):
        filename = files[0]

        if len(files) > 1:
            outname = files[1]
        else:
            outname = os.path.basename(filename)
            outname,ext = os.path.splitext(outname)
            outname = outname + '.preprocess' + ext
    else:
        print('Please specify input and optionally output filenames')
        exit(1)

    if options.verbose:
        print('In:', filename)
        print('Out:', outname)

    fram = Fram(dbname=options.db, dbhost=options.dbhost)

    image = fits.getdata(filename, -1)
    header = fits.getheader(filename, -1)

    #### Basic calibration
    darkname = fram.find_image('masterdark', header=header, debug=False, fix_path=True)
    flatname = fram.find_image('masterflat', header=header, debug=False, fix_path=True)

    if options.verbose:
        print('Dark:', darkname)
        print('Flat:', flatname)

    if darkname:
        dark = fits.getdata(darkname, -1)
    else:
        dcname = fram.find_image('dcurrent', header=header, debug=False, fix_path=True)
        biasname = fram.find_image('bias', header=header, debug=False, fix_path=True)
        if dcname and biasname:
            bias = fits.getdata(biasname, -1)
            dc = fits.getdata(dcname, -1)

            dark = bias + header['EXPOSURE']*dc
        else:
            dark = None

    image,header = calibrate.calibrate(image, header, dark=dark)

    if flatname:
        flat = fits.getdata(flatname, -1)

        image *= np.nanmedian(flat)/flat

    fits.writeto(outname, image, header, overwrite=True)
