#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import posixpath, glob, sys

from astropy.io import fits

from fram.resolve import resolve
from fram.fram import Fram

if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog [options] arg")
    # 'Object in the field' search
    parser.add_option('-o', '--object', help='Object name, to be visible on all frames', action='store', dest='object', type='str', default=None)

    # 'Frame center in the cone' search
    parser.add_option('--ra', help='Center RA', action='store', dest='ra', type='float', default=None)
    parser.add_option('--dec', help='Center Dec', action='store', dest='dec', type='float', default=None)
    parser.add_option('--sr', help='Search radius', action='store', dest='sr', type='float', default=None)

    # Refinement
    parser.add_option('-s', '--site', help='Site', action='store', dest='site', type='str', default=None)
    parser.add_option('-c', '--ccd', help='CCD', action='store', dest='ccd', type='str', default=None)
    parser.add_option('--serial', help='Camera serial number', action='store', dest='serial', type='int', default=None)
    parser.add_option('-t', '--target', help='Image target', action='store', dest='target', type='int', default=None)
    parser.add_option('-T', '--type', help='Image type', action='store', dest='type', type='str', default=None)
    parser.add_option('-f', '--filter', help='Filter', action='store', dest='filter', type='str', default=None)
    parser.add_option('-e', '--exposure', help='Exposure', action='store', dest='exposure', type='float', default=None)

    parser.add_option('-n', '--night', help='Night of observations', action='store', dest='night', type='str', default=None)
    parser.add_option('--night1', help='First night of observations', action='store', dest='night1', type='str', default=None)
    parser.add_option('--night2', help='Last night of observations', action='store', dest='night2', type='str', default=None)

    parser.add_option('--latest', help='Show latest images first', action='store_true', dest='latest', default=False)

    # Connection
    parser.add_option('-d', '--db', help='Database name', action='store', dest='db', type='str', default='fram')
    parser.add_option('-H', '--host', help='Database host', action='store', dest='dbhost', type='str', default=None)

    (options,args) = parser.parse_args()

    wheres,wargs = [],[]

    if options.object:
        target_name,target_ra,target_dec = resolve(options.object)

        if target_name:
            print('Object resolved to:', target_name, 'at', target_ra, target_dec, file=sys.stderr)
            wheres += ['q3c_radial_query(ra, dec, %s, %s, radius)']
            wargs += [target_ra, target_dec]
            wheres += ['q3c_poly_query(%s, %s, footprint10)']
            wargs += [target_ra, target_dec]
        else:
            print('Can\'t resolve:', object, file=sys.stderr)
            sys.exit(1)

    elif options.ra is not None and options.dec is not None and options.sr is not None:
        print('Searching for images with centers within', options.sr, 'deg around ', options.ra, options.dec, file=sys.stderr)
        wheres += ['q3c_radial_query(ra, dec, %s, %s, %s)']
        wargs += [options.ra, options.dec, options.sr]

    if options.site is not None:
        print('Searching for images from site', options.site, file=sys.stderr)
        wheres += ['site=%s']
        wargs += [options.site]

    if options.ccd is not None:
        print('Searching for images from ccd', options.ccd, file=sys.stderr)
        wheres += ['ccd=%s']
        wargs += [options.ccd]

    if options.serial is not None:
        print('Searching for images with serial', options.serial, file=sys.stderr)
        wheres += ['serial=%s']
        wargs += [options.serial]

    if options.target is not None:
        print('Searching for images with target', options.target, file=sys.stderr)
        wheres += ['target=%s']
        wargs += [options.target]

    if options.type is not None:
        print('Searching for images with type', options.type, file=sys.stderr)
        wheres += ['type=%s']
        wargs += [options.type]

    if options.filter is not None:
        print('Searching for images with filter', options.filter, file=sys.stderr)
        wheres += ['filter=%s']
        wargs += [options.filter]

    if options.exposure is not None:
        print('Searching for images with exposure', options.exposure, file=sys.stderr)
        wheres += ['exposure=%s']
        wargs += [options.exposure]

    if options.night is not None:
        print('Searching for images from night', options.night, file=sys.stderr)
        wheres += ['night=%s']
        wargs += [options.night]

    if options.night1 is not None:
        print('Searching for images night >=', options.night1, file=sys.stderr)
        wheres += ['night>=%s']
        wargs += [options.night1]

    if options.night2 is not None:
        print('Searching for images night <=', options.night2, file=sys.stderr)
        wheres += ['night<=%s']
        wargs += [options.night2]

    fram = Fram(dbname=options.db, dbhost=options.dbhost)

    if not fram:
        print('Can\'t connect to the database', file=sys.stderr)
        sys.exit(1)

    res = fram.query('SELECT filename FROM images WHERE ' + ' AND '.join(wheres) + ' ORDER BY time ' + ('DESC' if options.latest else 'ASC'), wargs)
    print(len(res), 'images found', file=sys.stderr)

    for r in res:
        print(r['filename'])
