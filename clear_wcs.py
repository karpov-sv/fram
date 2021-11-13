#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

from astropy.io import fits
import sys, glob, re

wcs_keywords = ['WCSAXES', 'CRPIX1', 'CRPIX2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CDELT1', 'CDELT2', 'CUNIT1', 'CUNIT2', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'LONPOLE', 'LATPOLE', 'RADESYS', 'EQUINOX', 'B_ORDER', 'A_ORDER', 'BP_ORDER', 'AP_ORDER', 'CD1_1', 'CD2_1', 'CD1_2', 'CD2_2', 'COMMENT', 'HISTORY', 'IMAGEW', 'IMAGEH']

if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    (options,args) = parser.parse_args()

    for filename in args:
        print(filename)

        fn = fits.open(filename, mode='update')
        header = fn[0].header

        for _ in header.keys():
            if _ and (_[0] == '_' or _ in wcs_keywords or re.match('^(A|B|AP|BP)_\d+_\d+$', _)):
                header.remove(_, remove_all=True, ignore_missing=True)

        fn.close()
