from __future__ import absolute_import, division, print_function, unicode_literals

from django.http import HttpResponse, FileResponse
from django.template.response import TemplateResponse
from django.shortcuts import redirect

from django.db.models import Count

import os, sys, posixpath
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from skimage.transform import rescale
from StringIO import StringIO
from esutil import htm

from astropy.io import fits
from astropy.wcs import WCS

from .models import Images
from .utils import permission_required_or_403
from . import settings

# FRAM modules
from .fram import calibrate
from .fram import survey
from .fram import utils
from .fram.fram import Fram

def get_images(request):
    images = Images.objects.all()

    night = request.GET.get('night')
    if night and night != 'all':
        images = images.filter(night=night)

    night1 = request.GET.get('night1')
    if night1:
        images = images.filter(night__gte=night1)

    night2 = request.GET.get('night2')
    if night2:
        images = images.filter(night__lte=night2)

    site = request.GET.get('site')
    if site and site != 'all':
        images = images.filter(site=site)

    fname = request.GET.get('filter')
    if fname and fname != 'all':
        images = images.filter(filter=fname)

    target = request.GET.get('target')
    if target and target != 'all':
        images = images.filter(target=target)

    tname = request.GET.get('type')
    if tname and tname != 'all':
        images = images.filter(type=tname)

    ccd = request.GET.get('ccd')
    if ccd and ccd != 'all':
        images = images.filter(ccd=ccd)

    serial = request.GET.get('serial')
    if serial and serial != 'all':
        images = images.filter(serial=serial)

    filename = request.GET.get('filename')
    if filename:
        if '%' in filename:
            # Extended syntax
            images = images.extra(where=["filename like %s"], params=(filename,))
        else:
            images = images.filter(filename__contains=filename)

    return images

def images_list(request):
    context = {}

    images = get_images(request)

    if request.GET.get('ra') and request.GET.get('dec'):
        ra = float(request.GET.get('ra'))
        dec = float(request.GET.get('dec'))
        sr = float(request.GET.get('sr', 0))
        context['ra'] = ra
        context['dec'] = dec
        context['sr'] = sr

        # Images with centers within given search radius
        images = images.extra(where=["q3c_radial_query(ra, dec, %s, %s, %s)"], params=(ra, dec, sr))

    # Possible values for fields
    types = images.distinct('type').values('type')
    context['types'] = types

    sites = images.distinct('site').values('site')
    context['sites'] = sites

    ccds = images.distinct('ccd').values('ccd')
    context['ccds'] = ccds

    filters = images.distinct('filter').values('filter')
    context['filters'] = filters

    sort = request.GET.get('sort')
    if sort:
        images = images.order_by(*(sort.split(',')))
    else:
        images = images.order_by('-time')

    context['images'] = images

    return TemplateResponse(request, 'images.html', context=context)

def images_cutouts(request):
    context = {}

    images = get_images(request)

    ra = float(request.GET.get('ra', 0))
    dec = float(request.GET.get('dec', 0))
    sr = float(request.GET.get('sr', 0.1))
    maxdist = float(request.GET.get('maxdist', 0.0))
    context['ra'] = ra
    context['dec'] = dec
    context['sr'] = sr
    context['maxdist'] = maxdist

    # Images containing given point
    images = images.extra(where=["q3c_radial_query(ra, dec, %s, %s, radius)"], params=(ra, dec))
    images = images.extra(select={'dist': "q3c_dist(ra, dec, %s, %s)"}, select_params=(ra,dec))
    images = images.extra(where=["q3c_poly_query(%s, %s, footprint10)"], params=(ra, dec))

    if maxdist > 0:
        images = images.extra(where=["q3c_dist(ra, dec, %s, %s) < %s"], params=(ra, dec, maxdist))

    # Possible values for fields
    sites = images.distinct('site').values('site')
    context['sites'] = sites

    ccds = images.distinct('ccd').values('ccd')
    context['ccds'] = ccds

    filters = images.distinct('filter').values('filter')
    context['filters'] = filters

    sort = request.GET.get('sort')
    if sort:
        images = images.order_by(*(sort.split(',')))
    else:
        images = images.order_by('-time')

    context['images'] = images

    return TemplateResponse(request, 'images_cutouts.html', context=context)

def image_details(request, id=0):
    context = {}

    image = Images.objects.get(id=id)
    context['image'] = image

    try:
        # Try to read original FITS keywords with comments
        filename = posixpath.join(settings.BASE_DIR, image.filename)
        header = fits.getheader(filename, -1)

        ignored_keywords = ['COMMENT', 'SIMPLE', 'BZERO', 'BSCALE', 'EXTEND', 'HISTORY']
        keywords = [{'key':k, 'value':repr(header[k]), 'comment':header.comments[k]} for k in header.keys() if k not in ignored_keywords]

        context['keywords'] = keywords
    except:
        pass

    return TemplateResponse(request, 'image.html', context=context)

def image_preview(request, id=0, size=0):
    image = Images.objects.get(id=id)
    filename = image.filename
    filename = posixpath.join(settings.BASE_DIR, filename)

    data = fits.getdata(filename, -1)
    header = fits.getheader(filename, -1)

    if not request.GET.has_key('raw'):
        data,header = calibrate.crop_overscans(data, header)

    if size:
        data = rescale(data, size/data.shape[1], mode='reflect', multichannel=False, anti_aliasing=True)

    figsize = (data.shape[1], data.shape[0])

    fig = Figure(facecolor='white', dpi=72, figsize=(figsize[0]/72, figsize[1]/72))

    limits = np.percentile(data, [0.5, 99.5])
    fig.figimage(data, vmin=limits[0], vmax=limits[1], origin='lower')

    canvas = FigureCanvas(fig)

    response = HttpResponse(content_type='image/jpeg')
    canvas.print_jpg(response)

    return response

def image_download(request, id):
    image = Images.objects.get(id=id)

    filename = image.filename
    filename = posixpath.join(settings.BASE_DIR, filename)

    response = HttpResponse(FileResponse(file(filename)), content_type='application/octet-stream')
    response['Content-Disposition'] = 'attachment; filename='+os.path.split(filename)[-1]
    response['Content-Length'] = os.path.getsize(filename)
    return response

def images_nights(request):
    # nights = Images.objects.order_by('-night').values('night').annotate(count=Count('night'))
    nights = Images.objects.values('night','site').annotate(count=Count('id')).order_by('-night','site')

    site = request.GET.get('site')
    if site and site != 'all':
        nights = nights.filter(site=site)

    context = {'nights':nights}

    sites = Images.objects.distinct('site').values('site')
    context['sites'] = sites

    return TemplateResponse(request, 'nights.html', context=context)

def image_fwhm(request, id=0):
    image = Images.objects.get(id=id)
    filename = image.filename
    filename = posixpath.join(settings.BASE_DIR, filename)

    data = fits.getdata(filename, -1)
    header = fits.getheader(filename, -1)

    data,header = calibrate.crop_overscans(data, header)

    fig = Figure(facecolor='white', dpi=72, figsize=(14,12))
    ax = fig.add_subplot(111)

    obj = survey.get_objects_sep(data, use_fwhm=True)
    utils.binned_map(obj['x'], obj['y'], obj['fwhm'], bins=16, statistic='median', ax=ax)
    ax.set_title('%s - %s %s %s %s - half flux radius mean %.2f median %.2f pix' % (posixpath.split(filename)[-1], image.site, image.ccd, image.filter, str(image.exposure), np.mean(obj['fwhm']), np.median(obj['fwhm'])))

    canvas = FigureCanvas(fig)

    response = HttpResponse(content_type='image/jpeg')
    canvas.print_jpg(response)

    return response

def image_wcs(request, id=0):
    image = Images.objects.get(id=id)
    filename = image.filename
    filename = posixpath.join(settings.BASE_DIR, filename)

    data = fits.getdata(filename, -1)
    header = fits.getheader(filename, -1)

    data,header = calibrate.crop_overscans(data, header)
    wcs = WCS(header)

    fig = Figure(facecolor='white', dpi=72, figsize=(14,12))
    ax = fig.add_subplot(111)

    # Detect objects
    obj = survey.get_objects_sep(data, use_fwhm=True, verbose=False)
    obj['ra'],obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)

    # Get stars from catalogue
    fram = Fram()
    ra0,dec0,sr0 = survey.get_frame_center(header=header)
    cat = fram.get_stars(ra0, dec0, sr0, extra=['v > 5 and v < 12'])
    x,y = wcs.all_world2pix(cat['ra'], cat['dec'], 0)

    # Match stars
    h = htm.HTM(10)
    m = h.match(obj['ra'],obj['dec'], cat['ra'],cat['dec'], 50.0/3600, maxmatch=0)
    oidx = m[0]
    cidx = m[1]
    dist = m[2]*3600

    utils.binned_map(obj['x'][oidx], obj['y'][oidx], dist, show_dots=True, bins=16, statistic='median', ax=ax)

    ax.set_title('%s - %s %s %s - displacement mean %.1f median %.1f arcsec' % (posixpath.split(filename)[-1], image.site, image.ccd, image.filter, np.mean(dist), np.median(dist)))

    canvas = FigureCanvas(fig)

    response = HttpResponse(content_type='image/jpeg')
    canvas.print_jpg(response)

    return response

def image_cutout(request, id=0, size=0, mode='view'):
    image = Images.objects.get(id=id)
    filename = image.filename
    filename = posixpath.join(settings.BASE_DIR, filename)

    data = fits.getdata(filename, -1)
    header = fits.getheader(filename, -1)
    data,header = calibrate.crop_overscans(data, header)

    ra,dec,sr = float(request.GET.get('ra')), float(request.GET.get('dec')), float(request.GET.get('sr'))

    wcs = WCS(header)
    x0,y0 = wcs.all_world2pix(ra, dec, sr)
    r0 = sr/np.hypot(wcs.pixel_scale_matrix[0,0], wcs.pixel_scale_matrix[0,1])

    crop,cropheader = utils.crop_image(data, x0, y0, r0, header)

    if mode == 'download':
        s = StringIO()
        fits.writeto(s, crop, cropheader)

        response = HttpResponse(s.getvalue(), content_type='application/octet-stream')
        response['Content-Disposition'] = 'attachment; filename=crop_'+os.path.split(filename)[-1]
        response['Content-Length'] = len(s.getvalue())
        return response

    if size:
        if size > crop.shape[1]:
            crop = rescale(crop, size/crop.shape[1], mode='reflect', multichannel=False, anti_aliasing=False, order=0)
        else:
            crop = rescale(crop, size/crop.shape[1], mode='reflect', multichannel=False, anti_aliasing=True)

    figsize = (crop.shape[1], crop.shape[0])

    fig = Figure(facecolor='white', dpi=72, figsize=(figsize[0]/72, figsize[1]/72))

    if np.any(np.isfinite(crop)):
        limits = np.percentile(crop[np.isfinite(crop)], [0.5, 99.0])

        fig.figimage(crop, vmin=limits[0], vmax=limits[1], origin='lower')

    canvas = FigureCanvas(fig)

    response = HttpResponse(content_type='image/jpeg')
    canvas.print_jpg(response)

    return response
