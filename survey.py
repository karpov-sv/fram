from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import posixpath, glob, datetime, os, sys, tempfile, shutil

from astropy.wcs import WCS
from astropy.io import fits

import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter(action='ignore', category=FITSFixedWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from esutil import coords, htm
import statsmodels.api as sm
from scipy.spatial import cKDTree

import sep,cv2
from StringIO import StringIO
import cPickle as pickle

def get_frame_center(filename=None, header=None, wcs=None, width=None, height=None):
    if not wcs:
        if header:
            wcs = WCS(header=header)
        elif filename:
            header = fits.getheader(filename, -1)
            wcs = WCS(header=header)

    if (not width or not height) and header:
        width = header['NAXIS1']
        height = header['NAXIS2']

    [ra1],[dec1] = wcs.all_pix2world([0], [0], 1)
    [ra0],[dec0] = wcs.all_pix2world([width/2], [height/2], 1)

    sr = coords.sphdist(ra0, dec0, ra1, dec1)[0]

    return ra0, dec0, sr

def blind_match_objects(obj, order=4, extra="", verbose=False):
    dir = tempfile.mkdtemp(prefix='astrometry')
    wcs = None
    binname = None
    ext = 0

    for path in ['.', '/usr/local', '/opt/local']:
        if os.path.isfile(posixpath.join(path, 'astrometry', 'bin', 'solve-field')):
            binname = posixpath.join(path, 'astrometry', 'bin', 'solve-field')
            break

    if binname:
        columns = [fits.Column(name='XIMAGE', format='1D', array=obj['x']+1),
                   fits.Column(name='YIMAGE', format='1D', array=obj['y']+1),
                   fits.Column(name='FLUX', format='1D', array=obj['flux'])]
        tbhdu = fits.BinTableHDU.from_columns(columns)
        filename = posixpath.join(dir, 'list.fits')
        tbhdu.writeto(filename, overwrite=True)
        extra += " --x-column XIMAGE --y-column YIMAGE --sort-column FLUX --width %d --height %d" % (np.ceil(max(obj['x']+1)), np.ceil(max(obj['y']+1)))

        wcsname = posixpath.split(filename)[-1]
        tmpname = posixpath.join(dir, posixpath.splitext(wcsname)[0] + '.tmp')
        wcsname = posixpath.join(dir, posixpath.splitext(wcsname)[0] + '.wcs')

        if verbose:
            print("%s -D %s --no-verify --overwrite --no-plots -T %s %s" % (binname, dir, extra, filename))
            return

        os.system("%s -D %s --no-verify --overwrite --no-plots -T %s %s" % (binname, dir, extra, filename))

        if order:
            order_str = "-t %d" % order
        else:
            order_str = "-T"

        if os.path.isfile(wcsname):
            shutil.move(wcsname, tmpname)
            os.system("%s -D %s --overwrite --no-plots %s %s --verify %s %s" % (binname, dir, order_str, extra, tmpname, filename))

            if os.path.isfile(wcsname):
                header = fits.getheader(wcsname)
                wcs = WCS(header)
    else:
        print("Astrometry.Net binary not found")

    #print order
    shutil.rmtree(dir)

    return wcs

def radectoxyz(ra_deg, dec_deg):
    ra  = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    xyz = np.array((np.cos(dec)*np.cos(ra),
                    np.cos(dec)*np.sin(ra),
                    np.sin(dec)))

    return xyz

def xyztoradec(xyz):
    ra = np.arctan2(xyz[1], xyz[0])
    ra += 2*np.pi * (ra < 0)
    dec = np.arcsin(xyz[2] / np.linalg.norm(xyz, axis=0))

    return (np.rad2deg(ra), np.rad2deg(dec))

def make_series(mul=1.0, x=1.0, y=1.0, order=1, sum=False, legendre=False, zero=True):
    if zero:
        res = [np.ones_like(x)*mul]
    else:
        res = []

    for i in xrange(1,order+1):
        maxr = i+1
        if legendre:
            maxr = order+1

        for j in xrange(maxr):
            #print i, '-', i - j, j
            if legendre:
                res.append(mul * leg(i)(x) * leg(j)(y))
            else:
                res.append(mul * x**(i-j) * y**j)
    if sum:
        return np.sum(res, axis=0)
    else:
        return res

def make_kernel(r0=1.0):
    x,y = np.mgrid[np.floor(-3.0*r0):np.ceil(3.0*r0+1), np.floor(-3.0*r0):np.ceil(3.0*r0+1)]
    r = np.hypot(x,y)
    image = np.exp(-r**2/2/r0**2)

    return image

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def get_objects_sep(image, header=None, mask=None, aper=3.0, bkgann=None, r0=0.5, gain=1, edge=0, minnthresh=2, minarea=5, relfluxradius=3.0, wcs=None, use_fwhm=False, verbose=True, get_fwhm75=False, get_fwhm90=False):
    if r0 > 0.0:
        kernel = make_kernel(r0)
    else:
        kernel = None

    if verbose:
        print("Preparing background mask")

    if mask is None:
        mask = np.zeros_like(image, dtype=np.bool)

    mask_bg = np.zeros_like(mask)
    mask_segm = np.zeros_like(mask)

    if False:
        # Simple heuristics to mask regions with rapidly varying background

        for _ in xrange(3):
            bg1 = sep.Background(image, mask=mask|mask_bg, bw=256, bh=256)
            bg2 = sep.Background(image, mask=mask|mask_bg, bw=32, bh=32)

            ibg = bg2.back() - bg1.back()

            tmp = np.abs(ibg - np.median(ibg)) > 5.0*1.4*mad(ibg)
            mask_bg |= cv2.dilate(tmp.astype(np.uint8), np.ones([100,100])).astype(np.bool)

    if verbose:
        print("Building background map")

    bg = sep.Background(image, mask=mask|mask_bg, bw=64, bh=64)
    image1 = image - bg.back()

    sep.set_extract_pixstack(image.shape[0]*image.shape[1])

    if False:
        # Mask regions around huge objects as they are most probably corrupted by saturation and blooming
        if verbose:
            print("Extracting initial objects")

        obj0,segm = sep.extract(image1, err=bg.rms(), thresh=4, minarea=3, mask=mask|mask_bg, filter_kernel=kernel, segmentation_map=True)

        if verbose:
            print("Dilating large objects")

        mask_segm = np.isin(segm, [_+1 for _,npix in enumerate(obj0['npix']) if npix > 100])
        mask_segm = cv2.dilate(mask_segm.astype(np.uint8), np.ones([10,10])).astype(np.bool)

    if verbose:
        print("Extracting final objects")

    obj0 = sep.extract(image1, err=bg.rms(), thresh=4, minarea=minarea, mask=mask|mask_bg|mask_segm, filter_kernel=kernel)

    if use_fwhm:
        # Estimate FHWM and use it to get optimal aperture size
        fwhm = 2.0*np.sqrt(np.hypot(obj0['a'], obj0['b'])*np.log(2))
        fwhm = sep.flux_radius(image1, obj0['x'], obj0['y'], relfluxradius*fwhm*np.ones_like(obj0['x']), 0.5, mask=mask)[0]
        fwhm = np.median(fwhm)

        aper = 1.5*fwhm

        if verbose:
            print("FWHM = %.2g, aperture = %.2g" % (fwhm, aper))

    # Windowed positional parameters are often biased in crowded fields, let's avoid them for now
    # xwin,ywin,flag = sep.winpos(image1, obj0['x'], obj0['y'], 2.0, mask=mask)
    xwin,ywin = obj0['x'], obj0['y']

    # Filter out objects too close to frame edges
    idx = (np.round(xwin) > edge) & (np.round(ywin) > edge) & (np.round(xwin) < image.shape[1]-edge) & (np.round(ywin) < image.shape[0]-edge)

    if minnthresh:
        idx &= (obj0['tnpix'] >= minnthresh)

    if verbose:
        print("Measuring final objects")

    flux,fluxerr,flag = sep.sum_circle(image1, xwin[idx], ywin[idx], aper, err=bg.rms(), gain=gain, mask=mask|mask_bg|mask_segm, bkgann=bkgann)
    # For debug purposes, let's make also the same aperture photometry on the background map
    bgflux,bgfluxerr,bgflag = sep.sum_circle(bg.back(), xwin[idx], ywin[idx], aper, err=bg.rms(), gain=gain, mask=mask|mask_bg|mask_segm)

    bgnorm = bgflux/np.pi/aper**2

    # Fluxes to magnitudes
    mag = -2.5*np.log10(flux)
    magerr = 2.5*np.log10(1.0 + fluxerr/flux)

    # better FWHM estimation
    # fwhm = 2.0*np.sqrt(np.hypot(obj0['a'][idx], obj0['b'][idx])*np.log(2))
    fwhm = sep.flux_radius(image1, xwin[idx], ywin[idx], relfluxradius*aper*np.ones_like(xwin[idx]), 0.5, mask=mask)[0]
    fwhm75 = fwhm if not get_fwhm75 else sep.flux_radius(image1, xwin[idx], ywin[idx], relfluxradius*aper*np.ones_like(xwin[idx]), 0.75, mask=mask)[0]
    fwhm90 = fwhm if not get_fwhm90 else sep.flux_radius(image1, xwin[idx], ywin[idx], relfluxradius*aper*np.ones_like(xwin[idx]), 0.9, mask=mask)[0]

    # Quality cuts
    fidx = (flux > 0) & (magerr < 0.1)

    if wcs is None and header is not None:
        # If header is provided, we may build WCS from it
        wcs = WCS(header)

    if wcs is not None:
        # If WCS is provided we may convert x,y to ra,dec
        ra,dec = wcs.all_pix2world(obj0['x'][idx], obj0['y'][idx], 0)
    else:
        ra,dec = np.zeros_like(obj0['x'][idx]),np.zeros_like(obj0['y'][idx])

    if verbose:
        print("All done")

    return {'x':xwin[idx][fidx], 'y':ywin[idx][fidx], 'flux':flux[fidx], 'fluxerr':fluxerr[fidx], 'mag':mag[fidx], 'magerr':magerr[fidx], 'flags':obj0['flag'][idx][fidx]|flag[fidx], 'ra':ra[fidx], 'dec':dec[fidx], 'bg':bgflux[fidx], 'bgnorm':bgnorm[fidx], 'fwhm':fwhm[fidx], 'fwhm75':fwhm75[fidx], 'fwhm90':fwhm90[fidx], 'aper':aper, 'bkgann':bkgann}

def match_objects(obj, cat, sr, fname='V', order=4, thresh=5.0, clim=None):
    x0,y0,width,height = np.mean(obj['x']), np.mean(obj['y']), np.max(obj['x'])-np.min(obj['x']), np.max(obj['y'])-np.min(obj['y'])

    # Match stars
    h = htm.HTM(10)
    m = h.match(obj['ra'],obj['dec'], cat['ra'],cat['dec'], sr, maxmatch=0)
    # m = h.match(obj['ra'],obj['dec'], cat['ra'],cat['dec'], 15.0/3600, maxmatch=0)
    oidx = m[0]
    cidx = m[1]
    dist = m[2]

    if fname == 'B':
        cmag,cmagerr = cat['B'], cat['Berr']
    elif fname == 'V':
        cmag,cmagerr = cat['V'], cat['Verr']
    elif fname == 'R':
        cmag,cmagerr = cat['R'], cat['Rerr']
    elif fname == 'I':
        cmag,cmagerr = cat['I'], cat['Ierr']
    elif fname == 'z':
        cmag,cmagerr = cat['z'], cat['zerr']

    if cmag is not None:
        cmag = cmag[cidx]
        cmagerr = cmagerr[cidx]
    else:
        print('Unsupported filter:', fname)
        return None

    if clim is not None:
        idx = cmag < clim

        oidx,cidx,dist,cmag,cmagerr = [_[idx] for _ in oidx,cidx,dist,cmag,cmagerr]

    x,y = obj['x'][oidx],obj['y'][oidx]
    omag,omagerr = obj['mag'][oidx],obj['magerr'][oidx]
    oflags = obj['flags'][oidx]

    x = (x - x0)*2/width
    y = (y - y0)*2/height

    tmagerr = np.hypot(cmagerr, omagerr)

    delta_mag = cmag - omag
    weights = 1.0/tmagerr**2

    X = make_series(1.0, x, y, order=order)

    X = np.vstack(X).T
    Y = delta_mag

    idx = (oflags == 0)

    for iter in range(3):
        if len(X[idx]) < 3:
            print("Fit failed - %d objects" % len(X[idx]))
            return None

        C = sm.WLS(Y[idx], X[idx], weights=weights[idx]).fit()

        YY = np.sum(X*C.params,axis=1)
        idx = (oflags == 0)
        if thresh and thresh > 0:
            idx &= (np.abs((Y-YY)/tmagerr) < thresh)

    x = (obj['x'] - x0)*2/width
    y = (obj['y'] - y0)*2/height

    X = make_series(1.0, x, y, order=order)
    X = np.vstack(X).T
    YY1 = np.sum(X*C.params,axis=1)

    mag = obj['mag'] + YY1

    return {
        # Matching indices and a distance in degrees
        'cidx':cidx, 'oidx':oidx, 'dist':dist,
        # Pixel coordinates of matched stars
        'x':obj['x'][oidx], 'y':obj['y'][oidx], 'oflags':oflags,
        # All catalogue magnitudes of matched stars
        'cB':cat['B'][cidx], 'cV':cat['V'][cidx], 'cR':cat['R'][cidx], 'cI':cat['I'][cidx],
        'cBerr':cat['Berr'][cidx], 'cVerr':cat['Verr'][cidx], 'cRerr':cat['Rerr'][cidx], 'cIerr':cat['Ierr'][cidx],
        # Catalogue magnitudes of matched stars in proper filter
        'cmag':cmag, 'cmagerr':cmagerr, 'tmagerr':tmagerr,
        # Model zero point for all objects, and their corrected magnitudes
        'mag0':YY1, 'mag':mag,
        'Y':Y, 'YY':YY,
        # Subset of matched stars used in the fits
        'idx':idx}

def fix_wcs(obj, cat, sr, header=None, maxmatch=1, order=6, fix=True):
    '''Get a refined WCS solution based on cross-matching of objects with catalogue on the sphere.
    Uses external 'fit-wcs' binary from Astrometry.Net suite'''

    if header is not None:
        width,height = header['NAXIS1'],header['NAXIS2']
        wcs = WCS(header)

        if wcs:
            obj['ra'],obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)
    else:
        width,height = int(np.max(obj['x'])),int(np.max(obj['y']))

    h = htm.HTM(10)
    oidx,cidx,dist = h.match(obj['ra'], obj['dec'], cat['ra'], cat['dec'], sr, maxmatch=1)

    dir = tempfile.mkdtemp(prefix='astrometry')
    wcs = None
    binname = None

    for path in ['.', '/usr/local', '/opt/local']:
        if os.path.isfile(posixpath.join(path, 'astrometry', 'bin', 'fit-wcs')):
            binname = posixpath.join(path, 'astrometry', 'bin', 'fit-wcs')
            break

    if binname:
        columns = [fits.Column(name='FIELD_X', format='1D', array=obj['x'][oidx] + 1),
                   fits.Column(name='FIELD_Y', format='1D', array=obj['y'][oidx] + 1),
                   fits.Column(name='INDEX_RA', format='1D', array=cat['ra'][cidx]),
                   fits.Column(name='INDEX_DEC', format='1D', array=cat['dec'][cidx])]
        tbhdu = fits.BinTableHDU.from_columns(columns)
        filename = posixpath.join(dir, 'list.fits')
        wcsname = posixpath.join(dir, 'list.wcs')

        tbhdu.writeto(filename, overwrite=True)

        os.system("%s -c %s -o %s -W %d -H %d -C -s %d" % (binname, filename, wcsname, width, height, order))

        if os.path.isfile(wcsname):
            header = fits.getheader(wcsname)
            wcs = WCS(header)

            if fix and wcs:
                obj['ra'],obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)

    else:
        print("Astrometry.Net binary not found")

    #print order
    shutil.rmtree(dir)

    return wcs

def fix_distortion(obj, cat, header=None, wcs=None, width=None, height=None, dr=3.0):
    '''Refines object sky coordinates based on cross-match with the catalogue in pixel space.
    Does not use any external tools but can't provide a fixed WCS solution'''
    if header:
        wcs = wcs.WCS(header)
        width = header['NAXIS1']
        height = header['NAXIS2']

    if not wcs or not width or not height:
        print("Nothing to fix")
        return

    kdo = cKDTree(np.array([obj['x'], obj['y']]).T)
    xc,yc = wcs.all_world2pix(cat['ra'], cat['dec'], 0)
    kdc = cKDTree(np.array([xc,yc]).T)

    m = kdo.query_ball_tree(kdc, dr)
    nm = np.array([len(_) for _ in m])

    # Distortions
    dx = np.array([obj['x'][_] - xc[m[_][0]] if nm[_] == 1 else 0 for _ in xrange(len(m))])
    dy = np.array([obj['y'][_] - yc[m[_][0]] if nm[_] == 1 else 0 for _ in xrange(len(m))])

    # Normalized coordinates
    x = (obj['x'] - width/2)*2.0/width
    y = (obj['y'] - height/2)*2.0/height

    X = make_series(1.0, x, y, order=6)

    X = np.vstack(X).T
    Yx = dx
    Yy = dy

    # Use only unique matches
    idx = (nm == 1) # & (distm < 2.0*np.std(distm))
    Cx = sm.WLS(Yx[idx], X[idx]).fit()
    Cy = sm.WLS(Yy[idx], X[idx]).fit()

    YYx = np.sum(X*Cx.params, axis=1)
    YYy = np.sum(X*Cy.params, axis=1)

    obj['ra'],obj['dec'] = wcs.all_pix2world(obj['x']-YYx, obj['y']-YYy, 0)

def load_results(filename):
    res = None
    with open(filename, 'r') as ff:
        res = pickle.load(ff)
    return res

def store_results(filename, obj):
    dirname = posixpath.split(filename)[0]

    try:
        os.makedirs(dirname)
    except:
        pass

    with open(filename, 'w') as ff:
        pickle.dump(obj, ff)

def store_wcs(filename, wcs):
    dirname = posixpath.split(filename)[0]

    try:
        os.makedirs(dirname)
    except:
        pass

    hdu = fits.PrimaryHDU(header=wcs.to_header(relax=True))
    hdu.writeto(filename, overwrite=True)
