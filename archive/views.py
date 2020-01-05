from __future__ import absolute_import, division, print_function, unicode_literals

from django.http import HttpResponse
from django.template.response import TemplateResponse
from django.shortcuts import redirect

import datetime, re
from db import DB

from resolve import resolve

from .models import Images
from .utils import permission_required_or_403, redirect_get

def index(request):
    context = {}

    return TemplateResponse(request, 'index.html', context=context)

def search(request):
    if request.method == 'POST':
        # Form submission handling

        params = {}

        if request.POST.get('coords') and request.POST.get('sr'):
            coords = request.POST.get('coords')
            sr = request.POST.get('sr')
            sr = float(sr) if sr else 1

            name, ra, dec = resolve(coords)
            print(name,ra,dec)

            if name:
                params['ra'] = ra
                params['dec'] = dec
                params['sr'] = sr

        elif request.POST.get('coords'):
            params['target'] = request.POST.get('coords')

        for _ in ['site', 'type', 'ccd', 'filter', 'night1', 'night2', 'serial']:
            if request.POST.get(_) and request.POST.get(_) != 'all':
                params[_] = request.POST.get(_)

        return redirect_get('images',  get=params)

    # No form submitted, just render a search form
    context = {}

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