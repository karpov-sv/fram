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
import calibrate

def images_list(request):
    context = {}

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

    if request.GET.get('ra') and request.GET.get('dec'):
        ra = float(request.GET.get('ra'))
        dec = float(request.GET.get('dec'))
        sr = float(request.GET.get('sr', 0))
        context['ra'] = ra
        context['dec'] = dec
        context['sr'] = sr

        if sr > 0:
            # Images with centers within given search radius
            images = images.extra(where=["q3c_radial_query(ra, dec, %s, %s, %s)"], params=(ra, dec, sr))
        else:
            # Images containing given point
            images = images.extra(where=["q3c_radial_query(ra, dec, %s, %s, radius)"], params=(ra, dec))

    # Possible values for fields
    # if tname and tname != 'all':
    #     types = Images.objects.distinct('type').values('type')
    # else:
    types = images.distinct('type').values('type')
    context['types'] = types

    # if site and site != 'all':
    #     sites = Images.objects.distinct('site').values('site')
    # else:
    sites = images.distinct('site').values('site')
    context['sites'] = sites

    # if ccd and ccd != 'all':
    #     ccds = Images.objects.distinct('ccd').values('ccd')
    # else:
    ccds = images.distinct('ccd').values('ccd')
    context['ccds'] = ccds

    # if fname and fname != 'all':
    #     filters = Images.objects.distinct('filter').values('filter')
    # else:
    filters = images.distinct('filter').values('filter')
    context['filters'] = filters

    sort = request.GET.get('sort')
    if sort:
        images = images.order_by(*(sort.split(',')))
    else:
        images = images.order_by('-time')

    context['images'] = images

    return TemplateResponse(request, 'images.html', context=context)

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

def image_preview(request, id=0, size=0, interpolation='nearest'):
    image = Images.objects.get(id=id)
    filename = image.filename
    filename = posixpath.join(settings.BASE_DIR, filename)

    data = fits.getdata(filename, -1)
    header = fits.getheader(filename, -1)

    if not request.GET.has_key('raw'):
        data,header = calibrate.crop_overscans(data, header)

    if size:
        data = rescale(data, 1.0*size/data.shape[1], mode='reflect', multichannel=False, anti_aliasing=True)

    figsize = (data.shape[1], data.shape[0])

    fig = Figure(facecolor='white', dpi=72, figsize=(1.0*figsize[0]/72, 1.0*figsize[1]/72))

    limits = np.percentile(data, [0.5, 99.5])
    fig.figimage(data, vmin=limits[0], vmax=limits[1])

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
