#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, glob

import numpy as np
import posixpath, glob, sys

from astropy.wcs import WCS
from astropy.io import fits

import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter(action='ignore', category=FITSFixedWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import survey, calibrate
import astroscrappy
import cPickle as pickle

from fram import Fram, get_night, parse_iso_time

def store_results(filename, obj):
    dirname = posixpath.split(filename)[0]

    try:
        os.makedirs(dirname)
    except:
        # import traceback
        # traceback.print_exc()
        pass

    with open(filename, 'w') as ff:
        pickle.dump(obj, ff)

def process_file(filename, night=None, site=None, fram=None, verbose=False, replace=False):
    if fram is None:
        fram = Fram()

    if site is None:
        # Simple heuristics to derive the site name
        for _ in ['auger', 'cta-n', 'cta-s0', 'cta-s1']:
            if _ in filename:
                site = _
                break

    header = fits.getheader(filename)

    if header['IMAGETYP'] != 'object':
        return

    ccd = header.get('CCD_NAME')
    fname = header.get('FILTER', 'unknown')
    time = parse_iso_time(header['DATE-OBS'])

    if fname not in ['B', 'V', 'R', 'I', 'z', 'N']:
        return

    if fname == 'N' and site == 'cta-n':
        effecive_fname = 'R'
    else:
        effecive_fname = fname

    if night is None:
        if header.get('LONGITUD') is not None:
            night = get_night(time, lon=header['LONGITUD'])
        else:
            night = get_night(time, site=site)

    dirname = 'photometry/%s/%s/%s' % (site, night, ccd)
    basename = posixpath.splitext(posixpath.split(filename)[-1])[0]
    basename = dirname + '/' + basename

    if not replace and posixpath.exists(basename + '.pickle'):
        return

    if verbose:
        print(filename, site, night, ccd, fname, effecive_fname)

    image = fits.getdata(filename).astype(np.double)

    # Basic calibration
    darkname = fram.find_image('masterdark', header=header, debug=False)
    flatname = fram.find_image('masterflat', header=header, debug=False)
    if not darkname or not flatname:
        store_results(basename+'.pickle', None)
        return

    dark = fits.getdata(darkname)
    flat = fits.getdata(flatname)

    image,header = calibrate.calibrate(image, header, dark=dark)
    image0 = image.copy()

    image *= np.median(flat)/flat

    # Basic masking
    mask = image > 50000
    mask |= dark > np.median(dark) + 10.0*np.std(dark)
#     print(np.sum(mask))

    cmask = np.zeros_like(mask)

    # WCS + catalogue
    wcs = WCS(header)
    pixscale = np.hypot(wcs.pixel_scale_matrix[0,0], wcs.pixel_scale_matrix[0,1])

    ra0,dec0,sr0 = survey.get_frame_center(header=header)
#     print(ra0, dec0, sr0, pixscale*3600)

    if 'WF' in header['CCD_NAME']:
        cat = fram.get_stars(ra0, dec0, sr0, catalog='pickles', extra=[], limit=100000)
    else:
        cat = fram.get_stars(ra0, dec0, sr0, catalog='atlas', extra=[], limit=100000)
    # print(len(cat['ra']), 'stars')

    # Cosmic rays
    obj0 = survey.get_objects_sep(image, mask=mask, wcs=wcs, minnthresh=3, edge=10, use_fwhm=True, verbose=False)
    gain = header.get('GAIN', 1.0)
    if gain > 100:
        gain /= 1000
    cmask,cimage = astroscrappy.detect_cosmics(image0, inmask=mask, gain=gain, readnoise=10, psffwhm=np.median(obj0['fwhm']), satlevel=50000, verbose=False)
    cimage /= gain

    # Object extraction
    obj = survey.get_objects_sep(image, mask=mask|cmask, wcs=wcs, edge=3, use_fwhm=True, verbose=False)

    # Effective limit
    lim = None
    sr = pixscale*np.median(obj['fwhm'])

    for iter in xrange(5):
        match = survey.match_objects(obj, cat, sr, fname=effecive_fname, clim=lim)
        if match is None:
            break
        lim = np.percentile(match['mag'], [95.0])[0] + 0.0

        if iter > 1:
            sr = 3.0*np.percentile(match['dist'], 65.0)

    if match is None:
        store_results(basename+'.pickle', None)
        return

    # print(lim, sr*3600)

    # Store results
    try:
        os.makedirs(dirname)
    except:
        pass

    store_results(basename+'.pickle', {'filename':filename,
                                       'site':site, 'night':night, 'ccd':ccd, 'filter':fname,
                                       'time':time,
                                       'ra':obj['ra'], 'dec':obj['dec'],
                                       'mag':match['mag'], 'magerr':obj['magerr'], 'flags':obj['flags'],
                                       'std':np.std((match['Y']-match['YY'])[match['idx']]),
                                       'nstars':len((match['Y']-match['YY'])[match['idx']])})

if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog [options] arg")
    parser.add_option('-d', '--db', help='Database name', action='store', dest='db', type='str', default='fram')
    parser.add_option('-H', '--host', help='Database host', action='store', dest='dbhost', type='str', default=None)
    parser.add_option('-r', '--replace', help='Replace already existing records in database', action='store_true', dest='replace', default=False)
    parser.add_option('-v', '--verbose', help='Verbose', action='store_true', dest='verbose', default=False)

    (options,files) = parser.parse_args()

    fram = Fram(dbname=options.db, dbhost=options.dbhost)

    for i,filename in enumerate(files):
        if len(files) > 1:
            print(i, '/', len(files), filename)
        process_file(filename, fram=fram, verbose=options.verbose, replace=options.replace)
