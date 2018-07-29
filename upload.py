#!/usr/bin/env python

import numpy as np
import posixpath, glob, datetime, os, sys

from astropy import wcs as pywcs
from astropy.io import fits as pyfits

from scipy.spatial import cKDTree

from esutil import coords
import sep, cv2
from esutil import htm, coords

import statsmodels.api as sm
from scipy.stats import binned_statistic_2d
from skimage.measure import block_reduce

import survey

from fram import Fram
#fram = Fram()

nthreads = 8

dirs = glob.glob('/mnt/data0/auger/*/*')
dirs += glob.glob('/mnt/data0/cta-s0/2*/*')

dirs.sort(reverse=True)

print len(dirs), "dirs"

def process_dir(dir):
    fram = Fram()

    night = posixpath.split(dir)[-1]

    print night, '/', dir
    files = glob.glob('%s/[0-9]*/WF*/*.fits' % dir)
    files += glob.glob('%s/darks/WF*/*.fits' % dir)
    files += glob.glob('%s/skyflats/WF*/*.fits' % dir)
    files.sort()

    res = fram.query('select filename from images where night=%s', (night,), simplify=False)
    filenames = [_['filename'] for _ in res]

    j = 0

    for j,filename in enumerate(files):
        if filename in filenames:
            continue
        try:
            header = pyfits.getheader(filename)
            image = pyfits.getdata(filename)

            header.remove('HISTORY', remove_all=True, ignore_missing=True)
            header.remove('COMMENT', remove_all=True, ignore_missing=True)

            type = header.get('IMAGETYP', 'unknown')

            if type == 'object':
                ra0,dec0 = header['OBJRA'], header['OBJDEC']

                wcs = pywcs.WCS(header)
                ra,dec = wcs.all_pix2world([0, header['NAXIS1'], 0.5*header['NAXIS1']], [0, header['NAXIS2'], 0.5*header['NAXIS2']], 0)
                radius = 0.5*coords.sphdist(ra[0], dec[0], ra[1], dec[1])[0]

                ra0,dec0 = ra[2],dec[2]
            else:
                # Should we really discard WCS for non-object frames?
                ra0,dec0,radius = 0,0,0

            target = header['TARGET']

            ccd = header['CCD_NAME']
            filter = header.get('FILTER', 'unknown')
            time = datetime.datetime.strptime(header['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f')

            exposure = header['EXPOSURE']

            mean = np.mean(image)
            median = np.median(image)

            keywords = dict(header)

            fram.query('INSERT INTO images (filename,night,time,target,type,filter,ccd,ra,dec,radius,exposure,width,height,mean,median,keywords) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (filename) DO NOTHING', (filename,night,time,target,type,filter,ccd,ra0,dec0,radius,exposure,header['NAXIS1'],header['NAXIS2'],mean,median,keywords))

            sys.stdout.write('\r  %d / %d - %s - %s %s %s %s' % (j, len(files), filename, night, ccd, type, filter))
            sys.stdout.flush()

        except KeyboardInterrupt:
                raise

        except:
            import traceback
            traceback.print_exc()
            pass

        #break

    if j:
        print



if nthreads > 0:
    import multiprocessing
    from functools import partial

    pool = multiprocessing.Pool(nthreads)
    # Make wrapper function to pass our arguments inside worker processes
    fn = partial(process_dir)
    pool.map(fn, dirs)

    pool.close()
    pool.join()
else:
    for dirname in dirs:
        process_dir(dirname)
