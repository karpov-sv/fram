#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os, glob, sys
import datetime

from astropy.io import fits

from fram.fram import Fram
from fram.calibrate import calibrate, crop_overscans
from fram.calibrate import calibration_configs, find_calibration_config
from fram.calibrate import rmean, rstd

try:
    from tqdm.auto import tqdm
except:
    from tqdm import tqdm

def get_next_month(night):
    t = datetime.datetime.strptime(night, '%Y%m%d')
    year,month,day = t.year, t.month, t.day

    day = 1
    month += 1
    if month > 12:
        year += 1
        month = 1

    return datetime.datetime(year, month, day).strftime('%Y%m%d')

if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog [options] arg")

    parser.add_option('-B', '--basedir', help='Base directory for output files', action='store', dest='basedir', type='str', default='calibrations')
    parser.add_option('-s', '--site', help='Site', action='store', dest='site', type='str', default=None)
    parser.add_option('-c', '--ccd', help='CCD', action='store', dest='ccd', type='str', default=None)
    parser.add_option('--serial', help='Camera serial number', action='store', dest='serial', type='int', default=None)
    parser.add_option('-t', '--target', help='Image target', action='store', dest='target', type='int', default=None)
    parser.add_option('-f', '--filter', help='Filter', action='store', dest='filter', type='str', default=None)
    parser.add_option('-b', '--binning', help='Binning', action='store', dest='binning', type='str', default=None)
    parser.add_option('-e', '--exposure', help='Exposure', action='store', dest='exposure', type='float', default=None)
    parser.add_option('-n', '--night', help='Night of observations', action='store', dest='night', type='str', default=None)
    parser.add_option('--night1', help='First night of observations', action='store', dest='night1', type='str', default=None)
    parser.add_option('--night2', help='Last night of observations', action='store', dest='night2', type='str', default=None)

    parser.add_option('-r', '--replace', help='Replace existing files', action='store_true', dest='replace', default=False)

    # Connection
    parser.add_option('-d', '--db', help='Database name', action='store', dest='db', type='str', default='fram')
    parser.add_option('-H', '--host', help='Database host', action='store', dest='dbhost', type='str', default=None)

    (options,args) = parser.parse_args()

    wheres,wargs = [],[]

    wheres += ["(type='dark' or type='zero')"]

    if options.site is not None:
        print('Searching for images from site', options.site, file=sys.stderr)
        wheres += ['site=%s']
        wargs += [options.site]

    if options.ccd is not None:
        print('Searching for images from ccd', options.ccd, file=sys.stderr)
        wheres += ['ccd=%s']
        wargs += [options.ccd]

    if options.serial is not None:
        print('Searching for images with serial', options.serial, file=sys.stderr)
        wheres += ['serial=%s']
        wargs += [options.serial]

    if options.target is not None:
        print('Searching for images with target', options.target, file=sys.stderr)
        wheres += ['target=%s']
        wargs += [options.target]

    if options.filter is not None:
        print('Searching for images with filter', options.filter, file=sys.stderr)
        wheres += ['filter=%s']
        wargs += [options.filter]

    if options.binning is not None:
        print('Searching for images with binning', options.binning, file=sys.stderr)
        wheres += ['binning=%s']
        wargs += [options.binning]

    if options.exposure is not None:
        print('Searching for images with exposure', options.exposure, file=sys.stderr)
        wheres += ['exposure=%s']
        wargs += [options.exposure]

    if options.night is not None:
        print('Searching for images from night', options.night, file=sys.stderr)
        wheres += ['night=%s']
        wargs += [options.night]

    if options.night1 is not None:
        print('Searching for images night >=', options.night1, file=sys.stderr)
        wheres += ['night>=%s']
        wargs += [options.night1]

    if options.night2 is not None:
        print('Searching for images night <=', options.night2, file=sys.stderr)
        wheres += ['night<=%s']
        wargs += [options.night2]

    fram = Fram(dbname=options.db, dbhost=options.dbhost)

    if not fram:
        print('Can\'t connect to the database', file=sys.stderr)
        sys.exit(1)

    res = fram.query('SELECT * FROM images WHERE ' + ' AND '.join(wheres) + ' ORDER BY time ', wargs)
    print(len(res), 'dark images found', file=sys.stderr)

    if not len(res):
        sys.exit(0)

    res.sort('time')

    # Prepare the data
    fids = np.arange(len(res))

    means,medians,exps,ccds,serials,times,nights,filenames,filters,exposures,targets,sites,widths,binnings = [np.array([__[_] for __ in res]) for _ in ['mean','median','exposure','ccd','serial','time','night','filename','filter','exposure', 'target', 'site', 'width', 'binning']]

    temps,airtemps,sunalts,moonalts,moondists,moonphases,ambtemps,imgids,biasavgs,dates,naxes1,naxes2 = [np.array([__['keywords'].get(_) for __ in res]) for _ in ['CCD_TEMP','CCD_AIR','SUN_ALT', 'MOONALT', 'MOONDIST', 'MOONPHA','AMBTEMP','IMGID','BIASAVG', 'DATE', 'NAXIS1', 'NAXIS2']]

    biasavgs = biasavgs.astype(np.double)

    # Isolate individual sequences for processing
    for cfg in calibration_configs:
        idx = (serials == cfg['serial']) & (binnings == cfg['binning'])
        if cfg.has_key('date-before'):
            idx &= dates < cfg['date-before']
        if cfg.has_key('date-after'):
            idx &= dates > cfg['date-after']
        if cfg.has_key('width'):
            idx &= widths == cfg['width']

        # FIXME: Some hard-coded quality cuts filters
        idx1 = idx \
            & (temps < -19) \
            & (means < cfg.get('means_max', 1000)) \
            & (means > cfg.get('means_min', 0))

        idx1 &= ((targets==21) & (sunalts < -18) & (moonphases > 20)) \
            | ((targets==1) & (sunalts < -10) & (moonphases > 20)) \
            | ((targets==2000) & (sunalts < -10) & (moonphases > 20)) \
            | ((targets==2) & (sunalts < -6)) \
            | (targets==20)

        if len(means[idx1]) < 10:
            continue

        # Simple filter on frame mean values
        if cfg.has_key('airtemp_a'):
            bias = airtemps*cfg['airtemp_a'] + cfg['airtemp_b']
        else:
            bias = np.zeros_like(means)

        bias[np.isfinite(biasavgs)] = biasavgs[np.isfinite(biasavgs)]

        for exp in np.unique(exposures[idx1]):
            eidx = exposures == exp
            mean = rmean((means-bias)[idx1 & eidx])
            std = rstd((means-bias)[idx1 & eidx])

            idx1[eidx] &= np.abs((means-bias)[eidx] - mean) < 3.0*std

        # print('serial', cfg['serial'], '-', len(means[idx]))

        for site in np.unique(sites[idx1]):
            for ccd in np.unique(ccds[idx1 & (sites == site)]):
                for fsize in np.unique(zip(naxes1[idx1 & (sites == site) & (ccds == ccd)], naxes2[idx1 & (sites == site) & (ccds == ccd)]), axis=0):

                    idx11 = idx1 & (sites == site) & (ccds == ccd) & (naxes1 == fsize[0]) & (naxes2 == fsize[1])
                    idx01 = idx & (sites == site) & (ccds == ccd) & (naxes1 == fsize[0]) & (naxes2 == fsize[1])

                    print(site, ccd, cfg['serial'], cfg['binning'], fsize, '-', np.sum(idx01), np.sum(idx11))

                    night1 = nights[idx01][0]
                    night2 = night1

                    print(night1, night2, nights[idx01][-1])

                    while True:
                        if night1 > nights[idx01][-1] or night2 > nights[idx01][-1]:
                            print('end')
                            break

                        night2 = get_next_month(night2)

                        idx2 = idx11 & (nights >= night1) & (nights < night2)
                        print(night1, night2, np.sum(idx2))

                        fcnts = np.unique(exposures[idx2], return_counts=True)

                        # Require at least 10 images for at least four different exposures
                        if np.sum(fcnts[1] > 10) >= 4:
                            basename = os.path.join(options.basedir, site, 'masterdarks', 'dark_%s_%s_%s_%s_%s_%s' % (site, ccd, cfg['serial'], night1, cfg['binning'], '%sx%s' % (fsize[0], fsize[1])))
                            print(basename, np.sum(idx2), fcnts[1])
                            night1 = night2

                            # Ensure output directory exists
                            try:
                                os.makedirs(os.path.dirname(basename))
                            except:
                                pass

                            # Use the header from first image
                            filename1 = filenames[idx2][0]
                            header1 = fits.getheader(filename1)

                            if cfg.has_key('airtemp_a') and cfg.has_key('airtemp_b'):
                                header1['AIRTEMPA'] = cfg['airtemp_a']
                                header1['AIRTEMPB'] = cfg['airtemp_b']

                            if header1.get('DATASEC'):
                                header1.pop('DATASEC')

                            darks = {}

                            # Iterate over different exposures
                            for exp in np.unique(exposures[idx2]):
                                darkname = basename + '_%s.fits' % exp
                                idx3 = idx2 & (exposures == exp)

                                if len(means[idx3]) < 6:
                                    print(exp, ':', len(means[idx3]), 'frames, skipping')
                                    # Can't make proper master dark from a few frames
                                    continue

                                if os.path.exists(darkname) and not options.replace:
                                    print(exp, ': loading existing file')
                                    sum = fits.getdata(darkname).astype(np.double)
                                    header1 = fits.getheader(darkname)
                                else:
                                    sum = None
                                    N = 0

                                    images = []

                                    for filename in tqdm(filenames[idx3], leave=False):
                                        image,header = fits.getdata(filename).astype(np.double), fits.getheader(filename)
                                        image,header = crop_overscans(image, header, cfg=cfg)

                                        if header.get('DATASEC0'):
                                            header1['DATASEC0'] = header.get('DATASEC0')

                                        images.append(image)

                                        if len(images) == 3:
                                            median = np.median(images, axis=0)
                                            images = []

                                            sum = sum + median if sum is not None else median
                                            N += 1

                                    sum /= N

                                    header1['NDARKS'] = np.sum(idx3)
                                    header1['NDARKMED'] = N

                                    print(exp, ':', darkname)

                                header1['EXPOSURE'] = exp
                                header1['IMAGETYP'] = 'masterdark'
                                fits.writeto(darkname, sum, header1, overwrite=True)

                                darks[exp] = {'dark': sum, 'exp': exp, 'header': header1.copy()}

                            if len(darks.keys()) < 2:
                                continue

                            if not os.path.exists(basename + '_bias.fits') or not os.path.exists(basename + '_dcurrent.fits') or options.replace:
                                # Compute bias and dark current maps
                                mdarks = np.array([darks[_]['dark'].flatten() for _ in sorted(darks.keys())])
                                exps = np.array(sorted(darks.keys()))
                                p = np.polyfit(exps, mdarks, 1)

                                bias = p[1].reshape(darks[exps[0]]['dark'].shape)
                                dcurrent = p[0].reshape(darks[exps[0]]['dark'].shape)

                                header1['EXPOSURE'] = 0

                                header1['IMAGETYP'] = 'bias'
                                fits.writeto(basename + '_bias.fits', bias, header1, overwrite=True)
                                print(basename + '_bias.fits')

                                header1['IMAGETYP'] = 'dcurrent'
                                fits.writeto(basename + '_dcurrent.fits', dcurrent, header1, overwrite=True)
                                print(basename + '_dcurrent.fits')
