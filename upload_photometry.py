#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, glob

import numpy as np
import posixpath, glob, sys

import cPickle as pickle

from StringIO import StringIO

from fram import Fram, get_night, parse_iso_time

def load_results(filename):
    res = None
    with open(filename, 'r') as ff:
        res = pickle.load(ff)
    return res

if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog [options] arg")
    parser.add_option('-d', '--db', help='Database name', action='store', dest='db', type='str', default='fram')
    parser.add_option('-H', '--host', help='Database host', action='store', dest='dbhost', type='str', default=None)
    parser.add_option('-r', '--replace', help='Replace already existing records in database', action='store_true', dest='replace', default=False)
    parser.add_option('-v', '--verbose', help='Verbose', action='store_true', dest='verbose', default=False)

    (options,files) = parser.parse_args()

    fram = Fram(dbname=options.db, dbhost=options.dbhost)
    fram.conn.autocommit = False
    cur = fram.conn.cursor()

    N = 0

    for i,filename in enumerate(files):
        if len(files) > 1:
            print(i, '/', len(files), filename)

        p = load_results(filename)

        if p is None:
            continue

        id = fram.query('SELECT id FROM images WHERE filename=%s', (p['filename'],), simplify=True)

        if fram.query('SELECT EXISTS(SELECT image FROM photometry WHERE image=%s);', (id,) , simplify=True):
            continue

        s = StringIO()

        for i in xrange(len(p['ra'])):
            print(id, p['time'], p['night'], p['site'], p['ccd'], p['filter'], p['ra'][i], p['dec'][i], p['mag'][i], p['magerr'][i], p['flags'][i], p['std'], p['nstars'], sep='\t', end='\n', file=s)

        s.seek(0)
        cur.copy_from(s, 'photometry', sep='\t', columns=['image', 'time', 'night', 'site', 'ccd', 'filter', 'ra', 'dec', 'mag', 'magerr', 'flags', 'std', 'nstars'], size=655350)

        N += 1

        if N % 100 == 0:
            fram.conn.commit()

    fram.conn.commit()

    print(N, 'files uploaded')
