import numpy as np
import posixpath, glob, datetime, os, sys, tempfile, shutil

from astropy import wcs as pywcs
from astropy.io import fits as pyfits

from esutil import coords

import sep,cv2

def get_frame_center(filename=None, header=None, wcs=None, width=None, height=None):
    if not wcs:
        if header:
            wcs = pywcs.WCS(header=header)
        elif filename:
            header = pyfits.getheader(filename, -1)
            wcs = pywcs.WCS(header=header)

    if (not width or not height) and header:
        width = header['NAXIS1']
        height = header['NAXIS2']

    [ra1],[dec1] = wcs.all_pix2world([0], [0], 1)
    [ra0],[dec0] = wcs.all_pix2world([width/2], [height/2], 1)

    sr = coords.sphdist(ra0, dec0, ra1, dec1)[0]

    return ra0, dec0, sr

def blind_match_objects(obj, order=4, extra=""):
    dir = tempfile.mkdtemp(prefix='astrometry')
    wcs = None
    binname = None
    ext = 0

    for path in ['.', '/usr/local', '/opt/local']:
        if os.path.isfile(posixpath.join(path, 'astrometry', 'bin', 'solve-field')):
            binname = posixpath.join(path, 'astrometry', 'bin', 'solve-field')
            break

    if binname:
        columns = [pyfits.Column(name='XIMAGE', format='1D', array=obj['x']+1),
                   pyfits.Column(name='YIMAGE', format='1D', array=obj['y']+1),
                   pyfits.Column(name='FLUX', format='1D', array=obj['flux'])]
        tbhdu = pyfits.BinTableHDU.from_columns(columns)
        filename = posixpath.join(dir, 'list.fits')
        tbhdu.writeto(filename, overwrite=True)
        extra += " --x-column XIMAGE --y-column YIMAGE --sort-column FLUX --width %d --height %d" % (np.ceil(max(obj['x']+1)), np.ceil(max(obj['y']+1)))

        wcsname = posixpath.split(filename)[-1]
        tmpname = posixpath.join(dir, posixpath.splitext(wcsname)[0] + '.tmp')
        wcsname = posixpath.join(dir, posixpath.splitext(wcsname)[0] + '.wcs')

        os.system("%s -D %s --no-verify --overwrite --no-fits2fits --no-plots --use-sextractor -t %d %s %s 2>/dev/null >/dev/null" % (binname, dir, order, extra, filename))

        if os.path.isfile(wcsname):
            shutil.move(wcsname, tmpname)
            os.system("%s -D %s --overwrite --no-fits2fits --no-plots -t %d %s --verify %s %s 2>/dev/null >/dev/null" % (binname, dir, order, extra, tmpname, filename))

            if os.path.isfile(wcsname):
                header = pyfits.getheader(wcsname)
                wcs = pywcs.WCS(header)
    else:
        print "Astrometry.Net binary not found"

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

def get_objects_sep(image, header=None, mask=None, aper=3.0, bkgann=None, r0=0.5, gain=1, edge=0, minnthresh=2, minarea=5, relfluxradius=3.0, use_fwhm=False, verbose=True):
    if r0 > 0.0:
        kernel = make_kernel(r0)
    else:
        kernel = None

    if verbose:
        print "Preparing background mask"

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
        print "Building background map"

    bg = sep.Background(image, mask=mask|mask_bg, bw=64, bh=64)
    image1 = image - bg.back()

    sep.set_extract_pixstack(image.shape[0]*image.shape[1])

    if False:
        # Mask regions around huge objects as they are most probably corrupted by saturation and blooming
        if verbose:
            print "Extracting initial objects"

        obj0,segm = sep.extract(image1, err=bg.rms(), thresh=4, minarea=3, mask=mask|mask_bg, filter_kernel=kernel, segmentation_map=True)

        if verbose:
            print "Dilating large objects"

        mask_segm = np.isin(segm, [_+1 for _,npix in enumerate(obj0['npix']) if npix > 100])
        mask_segm = cv2.dilate(mask_segm.astype(np.uint8), np.ones([10,10])).astype(np.bool)

    if verbose:
        print "Extracting final objects"

    obj0 = sep.extract(image1, err=bg.rms(), thresh=4, minarea=minarea, mask=mask|mask_bg|mask_segm, filter_kernel=kernel)

    if use_fwhm:
        # Estimate FHWM and use it to get optimal aperture size
        fwhm = 2.0*np.sqrt(np.hypot(obj0['a'], obj0['b'])*np.log(2))
        fwhm = sep.flux_radius(image1, obj0['x'], obj0['y'], relfluxradius*fwhm*np.ones_like(obj0['x']), 0.5, mask=mask)[0]
        fwhm = np.median(fwhm)

        aper = 1.5*fwhm

        if verbose:
            print "FWHM = %.2g, aperture = %.2g" % (fwhm, aper)

    # Windowed positional parameters are often biased in crowded fields, let's avoid them for now
    # xwin,ywin,flag = sep.winpos(image1, obj0['x'], obj0['y'], 2.0, mask=mask)
    xwin,ywin = obj0['x'], obj0['y']

    # Filter out objects too close to frame edges
    idx = (np.round(xwin) > edge) & (np.round(ywin) > edge) & (np.round(xwin) < image.shape[1]-edge) & (np.round(ywin) < image.shape[0]-edge)

    if minnthresh:
        idx &= (obj0['tnpix'] >= minnthresh)

    if verbose:
        print "Measuring final objects"

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

    # Quality cuts
    fidx = (flux > 0) & (magerr < 0.1)

    if header:
        # If header is provided, we may build WCS from it and convert x,y to ra,dec
        wcs = pywcs.WCS(header)
        ra,dec = wcs.all_pix2world(obj0['x'][idx], obj0['y'][idx], 0)
    else:
        #ra,dec = None,None
        ra,dec = np.zeros_like(obj0['x'][idx]),np.zeros_like(obj0['y'][idx])

    if verbose:
        print "All done"

    return {'x':xwin[idx][fidx], 'y':ywin[idx][fidx], 'flux':flux[fidx], 'fluxerr':fluxerr[fidx], 'mag':mag[fidx], 'magerr':magerr[fidx], 'flags':obj0['flag'][idx][fidx]|flag[fidx], 'ra':ra[fidx], 'dec':dec[fidx], 'bg':bgflux[fidx], 'bgnorm':bgnorm[fidx], 'fwhm':fwhm[fidx]}

def get_objects_cat(image, header=None, mask=None, cat=None, aper=3.0, bkgann=None, r0=0.5, gain=1, edge=0, use_fwhm=False, verbose=True):
    if r0 > 0.0:
        kernel = make_kernel(r0)
    else:
        kernel = None

    if verbose:
        print "Preparing background mask"

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
        print "Building background map"

    bg = sep.Background(image, mask=mask|mask_bg, bw=64, bh=64)
    image1 = image - bg.back()

    sep.set_extract_pixstack(image.shape[0]*image.shape[1])

    if False:
        # Mask regions around huge objects as they are most probably corrupted by saturation and blooming
        if verbose:
            print "Extracting initial objects"

        obj0,segm = sep.extract(image1, err=bg.rms(), thresh=4, minarea=3, mask=mask|mask_bg, filter_kernel=kernel, segmentation_map=True)

        if verbose:
            print "Dilating large objects"

        mask_segm = np.isin(segm, [_+1 for _,npix in enumerate(obj0['npix']) if npix > 100])
        mask_segm = cv2.dilate(mask_segm.astype(np.uint8), np.ones([10,10])).astype(np.bool)

    if verbose:
        print "Extracting final objects"

    obj0 = sep.extract(image1, err=bg.rms(), thresh=4, minarea=3, mask=mask|mask_bg|mask_segm, filter_kernel=kernel)

    if use_fwhm:
        # Estimate FHWM and use it to get optimal aperture size
        fwhm = 2.0*np.sqrt(np.hypot(obj0['a'], obj0['b'])*np.log(2))
        fwhm = np.median(fwhm)

        aper = 1.5*fwhm

        if verbose:
            print "FWHM = %.2g, aperture = %.2g" % (fwhm, aper)

    wcs = WCS(header)
    xwin,ywin = wcs.all_world2pix(cat['ra'], cat['dec'], 0)

    # Filter out objects too close to frame edges
    idx = (np.round(xwin) > edge) & (np.round(ywin) > edge) & (np.round(xwin) < image.shape[1]-edge) & (np.round(ywin) < image.shape[0]-edge)

    if verbose:
        print "Measuring final objects"

    flux,fluxerr,flag = sep.sum_circle(image1, xwin[idx], ywin[idx], aper, err=bg.rms(), gain=gain, mask=mask|mask_bg|mask_segm, bkgann=bkgann)
    # For debug purposes, let's make also the same aperture photometry on the background map
    bgflux,bgfluxerr,bgflag = sep.sum_circle(bg.back(), xwin[idx], ywin[idx], aper, err=bg.rms(), gain=gain, mask=mask|mask_bg|mask_segm)

    bgnorm = bgflux/np.pi/aper**2

    fwhm = 2.0*np.sqrt(np.hypot(obj['a'], obj['b'])*np.log(2))

    # Fluxes to magnitudes
    mag = -2.5*np.log10(flux)
    magerr = 2.5*np.log10(1.0 + fluxerr/flux)

    # Quality cuts
    fidx = (flux > 0) & (magerr < 0.1)

    if header:
        # If header is provided, we may build WCS from it and convert x,y to ra,dec
        wcs = pywcs.WCS(header)
        ra,dec = wcs.all_pix2world(obj0['x'][idx], obj0['y'][idx], 0)
    else:
        #ra,dec = None,None
        ra,dec = np.zeros_like(obj0['x'][idx]),np.zeros_like(obj0['y'][idx])

    if verbose:
        print "All done"

    return {'x':xwin[idx][fidx], 'y':ywin[idx][fidx], 'flux':flux[fidx], 'fluxerr':fluxerr[fidx], 'mag':mag[fidx], 'magerr':magerr[fidx], 'flags':obj0['flag'][idx][fidx]|flag[fidx], 'ra':ra[fidx], 'dec':dec[fidx], 'bg':bgflux[fidx], 'bgnorm':bgnorm[fidx], 'fwhm':fwhm[fidx]}
