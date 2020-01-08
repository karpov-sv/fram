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

from astropy.io import fits
from astropy.wcs import WCS

from .models import Images
from .utils import permission_required_or_403
from . import settings

# FRAM modules
from .fram import calibrate
from .fram import survey
from .fram import utils

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

wcs_keywords = ['NAXIS1', 'NAXIS2', 'WCSAXES', 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD2_1', 'CD1_2', 'CD2_2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CDELT1', 'CDELT2', 'CUNIT1', 'CUNIT2', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'LONPOLE', 'LATPOLE', 'RADESYS', 'EQUINOX', 'DATE-OBS', 'BP_0_0', 'BP_0_1', 'A_3_1', 'A_3_0', 'BP_0_4', 'BP_0_2', 'B_3_0', 'B_3_1', 'BP_1_2', 'BP_3_1', 'BP_3_0', 'BP_1_1', 'B_1_2', 'B_1_3', 'B_1_1', 'B_2_1', 'B_2_0', 'BP_2_1', 'B_2_2', 'BP_1_3', 'B_ORDER', 'A_ORDER', 'B_0_4', 'B_0_3', 'B_0_2', 'BP_0_3', 'A_4_0', 'BP_ORDER', 'AP_4_0', 'B_4_0', 'BP_4_0', 'AP_ORDER', 'BP_2_2', 'AP_3_0', 'AP_3_1', 'A_1_1', 'BP_2_0', 'A_1_3', 'A_1_2', 'A_0_4', 'AP_2_2', 'AP_2_1', 'AP_2_0', 'A_0_2', 'A_0_3', 'A_2_2', 'BP_1_0', 'A_2_0', 'A_2_1', 'AP_1_0', 'AP_1_1', 'AP_1_2', 'AP_1_3', 'AP_0_4', 'AP_0_1', 'AP_0_0', 'AP_0_3', 'AP_0_2']

def check_pos(image, ra, dec, edge=10):
    header = {_:image.keywords[_] for _ in image.keywords.keys() if _ in wcs_keywords}
    wcs = WCS(header)

    x,y = wcs.all_world2pix(ra, dec, 0)

    if x > edge and x < header['NAXIS1']-edge and y > edge and y < header['NAXIS2']-edge:
        return True
    else:
        return False

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

    if request.GET.get('exact') and request.GET.get('exact') != '0':
        # Exact WCS-based recheck whether the position is inside the frame
        # FIXME: extremely SLOW!
        images = [_ for _ in images if check_pos(_, ra, dec)]
        print(len(images))

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
    ax.set_title(posixpath.split(filename)[-1] + ' ' + image.site + ' ' + image.ccd + ' ' + image.filter + ' ' + str(image.exposure))

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
