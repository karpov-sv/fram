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
