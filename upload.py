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

from calibrate import crop_overscans

from fram import Fram
#fram = Fram()

nthreads = 8

def process_dir(dir, dbname='fram'):
    fram = Fram()

    night = posixpath.split(dir)[-1]

    print night, '/', dir
    files = glob.glob('%s/[0-9]*/*/*.fits' % dir)
    files += glob.glob('%s/darks/*/*.fits' % dir)
    files += glob.glob('%s/skyflats/*/*.fits' % dir)
    files.sort()

    res = fram.query('select filename from images where night=%s', (night,), simplify=False)
    filenames = [_['filename'] for _ in res]

    j = 0

    for j,filename in enumerate(files):
        if filename in filenames:
            continue
        if 'focusing' in filename:
            continue
        if 'bad' in filename:
            continue
        try:
            header = pyfits.getheader(filename)
            image = pyfits.getdata(filename)

            image,header = crop_overscans(image, header, subtract=False)

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
            serial = header['product_id']
            filter = header.get('FILTER', 'unknown')
            time = datetime.datetime.strptime(header['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f')

            site = None
            # Simple heuristics to derive the site name
            for _ in ['auger', 'cta-n', 'cta-s0', 'cta-s1']:
                if _ in filename:
                    site = _
                    break

            exposure = header['EXPOSURE']

            mean = np.mean(image)
            median = np.median(image)

            keywords = dict(header)

            fram.query('INSERT INTO images (filename,night,time,target,type,filter,ccd,serial,site,ra,dec,radius,exposure,width,height,mean,median,keywords) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (filename) DO NOTHING', (filename,night,time,target,type,filter,ccd,serial,site,ra0,dec0,radius,exposure,header['NAXIS1'],header['NAXIS2'],mean,median,keywords))

            sys.stdout.write('\r  %d / %d - %s - %s %s %s %s %s' % (j, len(files), filename, night, ccd, site, type, filter))
            sys.stdout.flush()

        except KeyboardInterrupt:
                raise

        except:
            import traceback
            print "Exception while processing", dir, filename
            traceback.print_exc()
            pass

        #break

    if j:
        print


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog [options] arg")
    parser.add_option('-n', '--nthreads', help='Number of threads to use', action='store', dest='nthreads', type='int', default=1)
    parser.add_option('-d', '--db', help='Database name', action='store', dest='db', type='str', default='fram')

    (options,args) = parser.parse_args()

    dirs = args

    if not dirs:
        dirs = glob.glob('/mnt/data0/auger/2*/*')
        dirs += glob.glob('/mnt/data0/cta-n/2*/*')
        dirs += glob.glob('/mnt/data2/cta-s0/2*/*')
        dirs += glob.glob('/mnt/data2/cta-s1/2*/*')

    dirs.sort(reverse=True)

    print len(dirs), "dirs"

    if options.nthreads > 1:
        import multiprocessing
        from functools import partial

        pool = multiprocessing.Pool(options.nthreads)
        # Make wrapper function to pass our arguments inside worker processes
        fn = partial(process_dir, dbname=options.db)
        pool.map(fn, dirs, 1)

        pool.close()
        pool.join()

    else:
        for dirname in dirs:
            process_dir(dirname)
