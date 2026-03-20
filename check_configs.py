#!/usr/bin/env python3

import datetime
import sys

from collections import Counter, defaultdict

from fram.fram import Fram
from fram.calibrate import calibration_configs, find_calibration_config


def parse_night(night):
    return datetime.datetime.strptime(night, '%Y%m%d').date()


def get_date_obs(row):
    date_obs = row['keywords'].get('DATE-OBS')
    if date_obs:
        return date_obs
    if row['time'] is not None:
        return row['time'].strftime('%Y-%m-%dT%H:%M:%S')
    return row['night']


def is_covered(row):
    cfg = find_calibration_config(
        serial=row['serial'],
        binning=row['binning'],
        width=row['width'],
        height=row['height'],
        date=get_date_obs(row),
    )
    return cfg is not None


def split_segments(rows, max_gap_days=45):
    rows = sorted(rows, key=lambda row: row['time'])
    if not rows:
        return []

    segments = []
    segment = [rows[0]]
    previous_night = parse_night(rows[0]['night'])

    for row in rows[1:]:
        current_night = parse_night(row['night'])
        if (current_night - previous_night).days > max_gap_days:
            segments.append(segment)
            segment = [row]
        else:
            segment.append(row)

        previous_night = current_night

    segments.append(segment)
    return segments


def summarize_sizes(rows):
    counts = Counter((row['width'], row['height']) for row in rows)
    return ', '.join('%sx%s(%d)' % (width, height, count) for (width, height), count in sorted(counts.items()))


def summarize_types(rows):
    counts = Counter(row['type'] for row in rows)
    return ', '.join('%s(%d)' % (type_name, count) for type_name, count in sorted(counts.items()))


def get_reason(row):
    same_camera = [
        cfg for cfg in calibration_configs
        if cfg.get('serial') == row['serial'] and cfg.get('binning') == row['binning']
    ]

    if not same_camera:
        return 'no serial/binning entry'

    return 'serial/binning entry exists, but date/size is outside configured coverage'


def sort_group_key(value):
    return tuple('' if part is None else str(part) for part in value)


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage='usage: %prog [options]')

    parser.add_option('-s', '--site', help='Site', action='store', dest='site', type='str', default=None)
    parser.add_option('-c', '--ccd', help='CCD', action='store', dest='ccd', type='str', default=None)
    parser.add_option('--serial', help='Camera serial number', action='store', dest='serial', type='int', default=None)
    parser.add_option('-b', '--binning', help='Binning', action='store', dest='binning', type='str', default=None)
    parser.add_option('-n', '--night', help='Night of observations', action='store', dest='night', type='str', default=None)
    parser.add_option('--night1', help='First night of observations', action='store', dest='night1', type='str', default=None)
    parser.add_option('--night2', help='Last night of observations', action='store', dest='night2', type='str', default=None)
    parser.add_option('--max-gap-days', help='Split segments when consecutive nights differ by more than this many days', action='store', dest='max_gap_days', type='int', default=45)

    parser.add_option('-d', '--db', help='Database name', action='store', dest='db', type='str', default='fram')
    parser.add_option('-H', '--host', help='Database host', action='store', dest='dbhost', type='str', default=None)

    (options, args) = parser.parse_args()

    wheres, wargs = [], []
    wheres += ["(type='dark' or type='zero')"]

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

    if options.binning is not None:
        print('Searching for images with binning', options.binning, file=sys.stderr)
        wheres += ['binning=%s']
        wargs += [options.binning]

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

    res = fram.query('SELECT * FROM images WHERE ' + ' AND '.join(wheres) + ' ORDER BY time ', wargs)
    print(len(res), 'dark images found', file=sys.stderr)

    if not len(res):
        sys.exit(0)

    grouped = defaultdict(list)
    for row in res:
        key = (row['site'], row['ccd'], row['serial'], row['binning'])
        grouped[key].append(row)

    uncovered_segments = []
    partial_segments = []

    for key in sorted(grouped, key=sort_group_key):
        site, ccd, serial, binning = key
        segments = split_segments(grouped[key], max_gap_days=options.max_gap_days)

        for segment in segments:
            coverage = [(row, is_covered(row)) for row in segment]
            uncovered = [row for row, covered in coverage if not covered]

            if not uncovered:
                continue

            entry = {
                'site': site,
                'ccd': ccd,
                'serial': serial,
                'binning': binning,
                'night1': segment[0]['night'],
                'night2': segment[-1]['night'],
                'frames': len(segment),
                'uncovered_frames': len(uncovered),
                'types': summarize_types(segment),
                'sizes': summarize_sizes(segment),
                'uncovered_sizes': summarize_sizes(uncovered),
                'reason': get_reason(uncovered[0]),
            }

            if len(uncovered) == len(segment):
                uncovered_segments.append(entry)
            else:
                partial_segments.append(entry)

    print('Segments checked:', sum(len(split_segments(rows, options.max_gap_days)) for rows in grouped.values()))
    print('Fully uncovered segments:', len(uncovered_segments))
    print('Partially uncovered segments:', len(partial_segments))

    if not uncovered_segments and not partial_segments:
        print('All checked dark-frame configurations are covered by calibration_configs.')
        sys.exit(0)

    if uncovered_segments:
        print('\nFully uncovered:')
        for entry in uncovered_segments:
            print(
                '  site=%s ccd=%s serial=%s binning=%s nights=%s..%s frames=%d types=%s sizes=%s reason=%s'
                % (
                    entry['site'],
                    entry['ccd'],
                    entry['serial'],
                    entry['binning'],
                    entry['night1'],
                    entry['night2'],
                    entry['frames'],
                    entry['types'],
                    entry['sizes'],
                    entry['reason'],
                )
            )

    if partial_segments:
        print('\nPartially uncovered:')
        for entry in partial_segments:
            print(
                '  site=%s ccd=%s serial=%s binning=%s nights=%s..%s uncovered=%d/%d sizes=%s uncovered_sizes=%s reason=%s'
                % (
                    entry['site'],
                    entry['ccd'],
                    entry['serial'],
                    entry['binning'],
                    entry['night1'],
                    entry['night2'],
                    entry['uncovered_frames'],
                    entry['frames'],
                    entry['sizes'],
                    entry['uncovered_sizes'],
                    entry['reason'],
                )
            )
