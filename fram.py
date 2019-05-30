#!/usr/bin/env python

import numpy as np

from astropy.io import fits as pyfits
from astropy import wcs as pywcs

import tempfile, datetime, posixpath, shutil, re, os
import ephem

from db import DB

class Fram(DB):
    def __init__(self, latitude=-35.4959, longitude=-69.4497, elevation=1430, **kwargs):
        DB.__init__(self, **kwargs)

        self.obs = ephem.Observer()
        self.obs.lat = latitude*ephem.pi/180.0
        self.obs.lon = longitude*ephem.pi/180.0
        self.obs.elevation = elevation

        self.moon = ephem.Moon()
        self.sun = ephem.Sun()

    def find_image(self, time=None, type='masterdark', ccd=None, exposure=None, width=None, height=None, header=None, debug=False, full=False):
        where = []
        order_string = ""
        opts = []

        if header is not None:
            # Guess some parameters from header
            time = parse_iso_time(header['DATE-OBS'])
            ccd = header.get('CCD_NAME', ccd)
            width = header.get('NAXIS1', width)
            height = header.get('NAXIS2', height)

        if type is not None:
            where.append('type=%s')
            opts.append(type)

        if ccd is not None:
            where.append('ccd=%s')
            opts.append(ccd)

        if exposure is not None:
            where.append('exposure=%s')
            opts.append(exposure)

        if width is not None:
            where.append('width=%s')
            opts.append(width)

        if height is not None:
            where.append('height=%s')
            opts.append(height)

        if time is not None:
            order_string = 'order by abs(extract(epoch from %s-time))'
            opts.append(time)

        where_string = " AND ".join(where)
        if where_string:
            where_string = "WHERE " + where_string

        res = self.query("SELECT * FROM images " + where_string + " " + order_string + " LIMIT 1;", opts, simplify=False, debug=debug)

        if full:
            return res[0] if len(res) else None
        else:
            return res[0]['filename'] if len(res) else None


def parse_iso_time(string, header=None):
    if header is not None:
        string = header['DATE-OBS']

    return datetime.datetime.strptime(string, '%Y-%m-%dT%H:%M:%S.%f')
