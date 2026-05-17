#!/usr/bin/env python

import os, glob, sys


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog [options] arg")

    parser.add_option('-p', '--photodir', help='Base dir to store photometry', action='store', dest='photodir', type='str', default='photometry')

    (options,args) = parser.parse_args()

    for filename in sys.stdin:
        filename = filename.strip()

        # Simple heuristics to derive the site name
        for _ in ['auger2', 'auger', 'cta-n', 'cta-s0', 'cta-s1']:
            if _ in filename:
                site = _
                break

        # Rough but fast checking of whether the file is already processed
        if os.path.exists(os.path.splitext(os.path.join(options.photodir, site, '/'.join(filename.split('/')[-4:])))[0] + '.parquet'):
            continue

        print(filename)
