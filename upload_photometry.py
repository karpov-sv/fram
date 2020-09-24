#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, glob

import numpy as np
import posixpath, glob, sys

from StringIO import StringIO

from fram import Fram, get_night, parse_iso_time
import survey

def touch(filename):
    with open(filename, 'a'):
        pass

    os.utime(filename, None)

if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog [options] arg")
    parser.add_option('-d', '--db', help='Database name', action='store', dest='db', type='str', default='fram')
    parser.add_option('-H', '--host', help='Database host', action='store', dest='dbhost', type='str', default=None)
    parser.add_option('-r', '--replace', help='Replace already existing records in database', action='store_true', dest='replace', default=False)
    parser.add_option('-i', '--ignore', help='Ignore upload status file', action='store_true', dest='ignore', default=False)
    parser.add_option('-v', '--verbose', help='Verbose', action='store_true', dest='verbose', default=False)

    (options,files) = parser.parse_args()

    fram = Fram(dbname=options.db, dbhost=options.dbhost)
    fram.conn.autocommit = False
    cur = fram.conn.cursor()

    N = 0

    s = StringIO()
    filenames = []

    table = 'photometry'

    for i,filename in enumerate(files):
        if not options.replace and not options.ignore and posixpath.exists(filename + '.upload'):
            # print('Skipping', filename, 'as upload file exists')
            continue

        obj = survey.load_objects(filename)

        if obj is None:
            if not options.replace:
                touch(filename + '.upload')
            continue

        if False and fram.query('SELECT EXISTS(SELECT time FROM ' + table + ' WHERE time=%s AND site=%s AND ccd=%s);', (obj['time'], obj['site'], obj['ccd']), simplify=True):
            if not options.replace:
                touch(filename + '.upload')

            # print('Skipping', filename, 'as already in DB')
            continue

        id = fram.query('SELECT id FROM images WHERE filename=%s', (obj['filename'],), simplify=True)

        if options.ignore and fram.query('SELECT EXISTS(SELECT image FROM ' + table + ' WHERE image=%s);', (id,) , simplify=True):
            continue

        if len(files) > 1:
            print(i, '/', len(files), filename, len(obj['ra']))
            sys.stdout.flush()

        for i in xrange(len(obj['ra'])):
            print(id, obj['time'], obj['night'], obj['site'], obj['ccd'], obj['filter'], obj['ra'][i], obj['dec'][i], obj['calib_mag'][i], obj['calib_magerr'][i], int(obj['flags'][i]), obj['std'], obj['nstars'], obj['fwhm'][i], sep='\t', end='\n', file=s)

        columns = ['image', 'time', 'night', 'site', 'ccd', 'filter', 'ra', 'dec', 'mag', 'magerr', 'flags', 'std', 'nstars', 'fwhm']

        filenames.append(filename)

        N += 1

        if N % 100 == 0:
            s.seek(0)
            cur.copy_from(s, table, sep='\t', columns=columns, size=65535000)

            print('committing...')
            fram.conn.commit()

            for fn in filenames:
                if not posixpath.exists(fn + '.upload'):
                    touch(fn + '.upload')

            s = StringIO()
            filenames = []


    if N > 0:
        s.seek(0)
        cur.copy_from(s, table, sep='\t', columns=columns, size=65535000)

        print('committing...')
        fram.conn.commit()

        for fn in filenames:
            if not posixpath.exists(fn + '.upload'):
                touch(fn + '.upload')

    print(N, '/', len(files), 'files uploaded')
