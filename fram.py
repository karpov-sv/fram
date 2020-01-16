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

    def find_image(type='masterdark', night=None, site=None, ccd=None, serial=None, exposure=None, cropped_width=None, cropped_height=None, filter=None, header=None, debug=False, full=False):
        '''Find the calibration image of given type not later than given night
        and suitable for a given chip and filter'''
        where = []
        opts = []

        # Guess some parameters from header
        if header is not None:
            if night is None:
                night = get_night(parse_iso_time(header.get('DATE-OBS')), lon=header.get('LONGITUD'), site=site)

            if ccd is None:
                ccd = header.get('CCD_NAME')

            if serial is None:
                serial = header.get('product_id')

            if cropped_width is None:
                cropped_width = header.get('NAXIS1')
            if cropped_height is None:
                cropped_height = header.get('NAXIS2')

            if filter is None:
                filter = header.get('FILTER')

            if exposure is None:
                exposure = header.get('EXPOSURE')

        # Construct the query
        if type is not None:
            where.append('type=%s')
            opts.append(type)

        if ccd is not None:
            where.append('ccd=%s')
            opts.append(ccd)

        if serial is not None:
            where.append('serial=%s')
            opts.append(serial)

        if exposure is not None:
            where.append('exposure=%s')
            opts.append(exposure)

        if cropped_width is not None:
            where.append('cropped_width=%s')
            opts.append(cropped_width)

        if cropped_height is not None:
            where.append('cropped_height=%s')
            opts.append(cropped_height)

        if night is not None:
            where.append('night<=%s')
            opts.append(night)

        where_string = " AND ".join(where)
        if where_string:
            where_string = "WHERE " + where_string

        res = self.query("SELECT * FROM calibrations " + where_string + " ORDER BY night DESC LIMIT 1;", opts, simplify=False, debug=debug)

        if full:
            return res[0] if len(res) else None
        else:
            return res[0]['filename'] if len(res) else None


def parse_iso_time(string, header=None):
    if header is not None:
        string = header['DATE-OBS']

    return datetime.datetime.strptime(string, '%Y-%m-%dT%H:%M:%S.%f')

def get_night(time, lon=None, site=None):
    if lon is None:
        if site == 'auger':
            lon = -69.4497
        elif site == 'cta-n':
            lon = -17.89
        elif site == 'cta-s0' or site == 'cta-s1':
            lon = -70.32482
        else:
            lon = 0

    time1 = time + datetime.timedelta(seconds=lon*86400/360 - 86400/2)

    return time1.strftime('%Y%m%d')
