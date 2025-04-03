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

from fram import survey, calibrate
import astroscrappy
#import cPickle as pickle
import pickle

from fram.fram import Fram, get_night, parse_iso_time
from fram.match import Match

def process_file(filename, night=None, site=None, fram=None, verbose=False, replace=False, base='photometry'):
    if not posixpath.exists(filename):
        return None

    if site is None:
        # Simple heuristics to derive the site name
        for _ in ['auger', 'cta-n', 'cta-s0', 'cta-s1']:
            if _ in filename:
                site = _
                break

    # Rough but fast checking of whether the file is already processed
    if not replace and posixpath.exists(posixpath.splitext(posixpath.join(base, site, '/'.join(filename.split('/')[-4:])))[0] + '.cat'):
        return

    header = fits.getheader(filename)

    if header['IMAGETYP'] != 'object':
        return

    ccd = header.get('CCD_NAME')
    fname = header.get('FILTER', 'unknown')
    time = parse_iso_time(header['DATE-OBS'])
    target = header.get('TARGET', -1)

    if fname not in ['B', 'V', 'R', 'I', 'z', 'N']:
        return

    if fname == 'N' and site == 'cta-n':
        effective_fname = 'R'
    else:
        effective_fname = fname

    if night is None:
        if header.get('LONGITUD') is not None:
            night = get_night(time, lon=header['LONGITUD'])
        else:
            night = get_night(time, site=site)

    dirname = '%s/%s/%s/%05d/%s' % (base, site, night, target, ccd)
    basename = posixpath.splitext(posixpath.split(filename)[-1])[0]
    basename = dirname + '/' + basename
    catname = basename + '.cat'

    if not replace and posixpath.exists(catname):
        return

    if verbose:
        print(filename, site, night, ccd, fname, effective_fname)

    image = fits.getdata(filename).astype(np.double)

    if fram is None:
        fram = Fram()

    # Basic calibration
    darkname = fram.find_image('masterdark', header=header, debug=False)
    flatname = fram.find_image('masterflat', header=header, debug=False)

    if darkname:
        dark = fits.getdata(darkname)
    else:
        dcname = fram.find_image('dcurrent', header=header, debug=False)
        biasname = fram.find_image('bias', header=header, debug=False)
        if dcname and biasname:
            bias = fits.getdata(biasname)
            dc = fits.getdata(dcname)

            dark = bias + header['EXPOSURE']*dc
        else:
            dark = None

    if flatname:
        flat = fits.getdata(flatname)
    else:
        flat = None

    if dark is None or flat is None:
        survey.save_objects(catname, None)
        return

    image,header = calibrate.calibrate(image, header, dark=dark)
    image0 = image.copy()

    image *= np.median(flat)/flat

    # Basic masking
    mask = image > 50000
    mask |= dark > np.median(dark) + 10.0*np.std(dark)

    cmask = np.zeros_like(mask)

    # WCS + catalogue
    wcs = WCS(header)
    pixscale = np.hypot(wcs.pixel_scale_matrix[0,0], wcs.pixel_scale_matrix[0,1])
    gain = header.get('GAIN', 1.0)
    if gain > 100:
        gain /= 1000

    ra0,dec0,sr0 = survey.get_frame_center(header=header)

    if 'WF' in header['CCD_NAME']:
        if header['CCD_NAME'] in ['WF6', 'WF7', 'WF8']:
            cat = fram.get_stars(ra0, dec0, sr0, limit=100000, catalog='gaia', extra=['g<15', 'good=1 and var=0 and multi_30=0'])
        else:
            cat = fram.get_stars(ra0, dec0, sr0, limit=100000, catalog='gaia', extra=['g<15', 'good=1 and var=0 and multi_70=0'])

    else:
        cat = fram.get_stars(ra0, dec0, sr0, catalog='atlas', extra=[], limit=100000)

    # Cosmic rays
    if not 'WF' in header['CCD_NAME']:
        obj0 = survey.get_objects_sep(image, mask=mask, wcs=wcs, minnthresh=3, edge=10, use_fwhm=True, sn=10, verbose=False)
        cmask,cimage = astroscrappy.detect_cosmics(image0, inmask=mask, gain=gain, readnoise=10, psffwhm=np.median(obj0['fwhm']), satlevel=50000, verbose=False)
        cimage /= gain

    # Object extraction
    if ccd == 'C0':
        obj = survey.get_objects_sep(image, mask=mask|cmask, wcs=wcs, edge=10, aper=5, verbose=False, sn=5)

    else:
        obj = survey.get_objects_sextractor(image, mask=mask|cmask, wcs=wcs, gain=gain, edge=10, aper=3.0, minarea=3.0, r0=0, sn=3, verbose=False, _tmpdir='tmp/', extra_params=['FLUX_MAX'])

    # Match with catalogue
    match = Match(width=image.shape[1], height=image.shape[0])
    sr = pixscale*np.median(obj['fwhm'])

    if not match.match(obj=obj, cat=cat, sr=sr, filter_name=effective_fname, order=0, bg_order=None, color_order=None, verbose=False) or match.ngoodstars < 10:
        # if verbose:
        #     print(match.ngoodstars, 'good matches, retrying without spatial term')

        # if not match.match(obj=obj, cat=cat, sr=sr, filter_name=effective_fname, order=0, bg_order=None, color_order=None, verbose=False) or match.ngoodstars < 10:
        if verbose:
            print('Matching failed for', filename, ':', match.ngoodstars, 'good matches')

        survey.save_objects(catname, None)
        return

    if verbose:
        print(match.ngoodstars, 'good matches, std =', match.std)

    # Store results
    try:
        os.makedirs(dirname)
    except:
        pass

    obj['mag_limit'] = match.mag_limit
    obj['color_term'] = match.color_term

    obj['filename'] = filename
    obj['site'] = site
    obj['night'] = night
    obj['ccd'] = ccd
    obj['filter'] = fname
    obj['cat_filter'] = effective_fname
    obj['time'] = time

    obj['mag_id'] = match.mag_id

    obj['good_idx'] = match.good_idx
    obj['calib_mag'] = match.mag
    obj['calib_magerr'] = match.magerr

    obj['std'] = match.std
    obj['nstars'] = match.ngoodstars

    survey.save_objects(catname, obj, header=header)

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
        try:
            process_file(filename, fram=fram, verbose=options.verbose, replace=options.replace)
        except KeyboardInterrupt:
            raise
        except:
            print('\nException while processing:', filename, file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise
