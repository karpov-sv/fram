from __future__ import absolute_import, division, print_function, unicode_literals

from django.http import HttpResponse
from django.template.response import TemplateResponse
from django.shortcuts import redirect

import datetime, re

from .models import Images
from .utils import permission_required_or_403, redirect_get, db_query

# FRAM modules
from .fram.resolve import resolve

def index(request):
    context = {}

    sites = db_query('select site,count(*),(select night from images where site=i.site order by time desc limit 1) as last, (select night from images where site=i.site order by time asc limit 1) as first from images i group by i.site order by i.site;', (), simplify=False)

    context['sites'] = sites

    return TemplateResponse(request, 'index.html', context=context)

def search(request, mode='images'):
    message,message_cutout = None,None

    if request.method == 'POST':
        # Form submission handling

        params = {}

        for _ in ['site', 'type', 'ccd', 'filter', 'night1', 'night2', 'serial', 'target', 'maxdist', 'filename']:
            if request.POST.get(_) and request.POST.get(_) != 'all':
                params[_] = request.POST.get(_)

        if mode == 'cutouts':
            # Search cutouts only
            coords = request.POST.get('coords')
            sr = request.POST.get('sr')
            name,ra,dec = resolve(coords)

            if name:
                params['name'] = name
                params['ra'] = ra
                params['dec'] = dec
                params['sr'] = float(sr) if sr else 0.1

                return redirect_get('images_cutouts',  get=params)
            else:
                message_cutout = "Can't resolve the query position: " + coords

        else:
            # Search full images
            if request.POST.get('coords') and not request.POST.get('sr'):
                message = "No search radius specified"
            elif request.POST.get('coords') and request.POST.get('sr'):
                coords = request.POST.get('coords')
                sr = request.POST.get('sr')
                sr = float(sr) if sr else 1

                name, ra, dec = resolve(coords)
                print(name,ra,dec)

                if name:
                    params['ra'] = ra
                    params['dec'] = dec
                    params['sr'] = sr

                    return redirect_get('images',  get=params)
                else:
                    message = "Can't resolve the query center: " + coords
            else:
                return redirect_get('images',  get=params)

    # No form submitted, just render a search form
    context = {'message': message, 'message_cutout': message_cutout}

    # Possible values for fields
    types = Images.objects.distinct('type').values('type')
    context['types'] = types

    sites = Images.objects.distinct('site').values('site')
    context['sites'] = sites

    ccds = Images.objects.distinct('ccd').values('ccd')
    context['ccds'] = ccds

    serials = Images.objects.distinct('serial').values('serial')
    context['serials'] = serials

    filters = Images.objects.distinct('filter').values('filter')
    context['filters'] = filters

    return TemplateResponse(request, 'search.html', context=context)
