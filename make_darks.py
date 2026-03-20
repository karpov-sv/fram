#!/usr/bin/env python3

import datetime
import os
import sys
import threading
import traceback

import numpy as np

from astropy.io import fits

from fram.fram import Fram
from fram.calibrate import calibration_configs, crop_overscans
from fram.calibrate import rmean, rstd

try:
    from tqdm.auto import tqdm
except:
    from tqdm import tqdm

from stdpipe.utils import fits_write

PROGRESS_QUEUE = None


def get_next_month(night):
    t = datetime.datetime.strptime(night, '%Y%m%d')
    year, month = t.year, t.month

    month += 1
    if month > 12:
        year += 1
        month = 1

    return datetime.datetime(year, month, 1).strftime('%Y%m%d')


def to_python_scalar(value):
    return value.item() if hasattr(value, 'item') else value


def init_worker(progress_queue):
    global PROGRESS_QUEUE
    PROGRESS_QUEUE = progress_queue


def emit_progress(amount):
    if PROGRESS_QUEUE is not None and amount:
        PROGRESS_QUEUE.put(amount)


def progress_consumer(progress_queue, progress_bar):
    while True:
        amount = progress_queue.get()
        if amount is None:
            break
        progress_bar.update(amount)


def build_masterdark(filenames, cfg, header_template, show_progress=False):
    sum_image = None
    nmedians = 0
    nused = 0
    images = []
    header_out = header_template.copy()
    progress_buffer = 0

    for filename in tqdm(filenames, leave=False, disable=not show_progress):
        image = fits.getdata(filename, -1).astype(np.double)
        header = fits.getheader(filename, -1)
        image, header = crop_overscans(image, header, cfg=cfg)

        if header.get('DATASEC0'):
            header_out['DATASEC0'] = header.get('DATASEC0')

        images.append(image)
        progress_buffer += 1

        if progress_buffer >= 10:
            emit_progress(progress_buffer)
            progress_buffer = 0

        if len(images) == 3:
            nimages = len(images)
            median = np.median(images, axis=0)
            images = []

            sum_image = nimages * median if sum_image is None else sum_image + nimages * median
            nmedians += 1
            nused += nimages

    if len(images):
        nimages = len(images)
        median = np.median(images, axis=0)
        sum_image = nimages * median if sum_image is None else sum_image + nimages * median
        nmedians += 1
        nused += nimages

    if progress_buffer:
        emit_progress(progress_buffer)

    if not nused:
        raise RuntimeError('No dark frames were stacked')

    sum_image /= nused

    header_out['NDARKS'] = nused
    header_out['NDARKMED'] = nmedians

    return sum_image, header_out


def process_segment(task):
    try:
        return _process_segment(task)
    except KeyboardInterrupt:
        raise
    except:
        return {
            'status': 'error',
            'basename': task['basename'],
            'traceback': traceback.format_exc(),
        }


def _process_segment(task):
    basename = task['basename']
    bias_name = basename + '_bias.fits'
    dcurrent_name = basename + '_dcurrent.fits'
    need_derived = task['need_derived']

    os.makedirs(os.path.dirname(basename), exist_ok=True)

    header_template = fits.getheader(task['header_filename'], -1)
    if 'airtemp_a' in task['cfg'] and 'airtemp_b' in task['cfg']:
        header_template['AIRTEMPA'] = task['cfg']['airtemp_a']
        header_template['AIRTEMPB'] = task['cfg']['airtemp_b']
    if header_template.get('DATASEC'):
        header_template.pop('DATASEC')

    darks = {}
    skipped = []

    for group in task['exposure_groups']:
        exp = group['exposure']
        filenames = group['filenames']
        dark_name = basename + '_%s.fits' % exp
        write_dark = True

        if len(filenames) < 6:
            skipped.append((exp, len(filenames)))
            continue

        if os.path.exists(dark_name) and not task['replace']:
            if not need_derived:
                continue

            dark = fits.getdata(dark_name, -1).astype(np.double)
            existing_header = fits.getheader(dark_name, -1)
            header = header_template.copy()
            if existing_header.get('DATASEC0'):
                header['DATASEC0'] = existing_header.get('DATASEC0')
            emit_progress(group['progress_total'])
            write_dark = False
        else:
            dark, header = build_masterdark(
                filenames,
                task['cfg'],
                header_template,
                show_progress=task['show_progress'],
            )

        header = header.copy()
        header['EXPOSURE'] = exp
        header['IMAGETYP'] = 'masterdark'
        if write_dark:
            fits_write(dark_name, dark, header, compress=True)

        darks[exp] = {'dark': dark, 'header': header}

    wrote_bias = False
    wrote_dcurrent = False

    if need_derived and len(darks) >= 2:
        sorted_exposures = sorted(darks.keys())
        exposure_values = np.array(sorted_exposures, dtype=np.double)
        stacked_darks = np.array([darks[exp]['dark'].ravel() for exp in sorted_exposures])
        coeffs = np.polyfit(exposure_values, stacked_darks, 1)

        bias = coeffs[1].reshape(darks[sorted_exposures[0]]['dark'].shape)
        dcurrent = coeffs[0].reshape(darks[sorted_exposures[0]]['dark'].shape)

        header = darks[sorted_exposures[0]]['header'].copy()
        header['EXPOSURE'] = 0

        header['IMAGETYP'] = 'bias'
        fits_write(bias_name, bias, header, compress=True)
        wrote_bias = True
        emit_progress(1)

        header['IMAGETYP'] = 'dcurrent'
        fits_write(dcurrent_name, dcurrent, header, compress=True)
        wrote_dcurrent = True
        emit_progress(1)

    return {
        'status': 'ok',
        'basename': basename,
        'n_selected': task['n_selected'],
        'n_darks': len(darks),
        'skipped': skipped,
        'wrote_bias': wrote_bias,
        'wrote_dcurrent': wrote_dcurrent,
    }


def report_result(result):
    if result['status'] == 'error':
        tqdm.write('ERROR while processing %s' % result['basename'], file=sys.stderr)
        tqdm.write(result['traceback'], file=sys.stderr)
        return False

    msg = '%s : %d masterdarks from %d selected frames' % (
        result['basename'],
        result['n_darks'],
        result['n_selected'],
    )

    if result['wrote_bias'] or result['wrote_dcurrent']:
        msg += ' [bias/dcurrent updated]'

    if result['skipped']:
        msg += ' [skipped %d sparse exposures]' % len(result['skipped'])

    tqdm.write(msg)
    return True


def build_tasks(res, options):
    res.sort('time')

    means, medians, exps, ccds, serials, times, nights, filenames, filters, exposures, targets, sites, widths, binnings = [
        np.array([row[key] for row in res])
        for key in ['mean', 'median', 'exposure', 'ccd', 'serial', 'time', 'night', 'filename', 'filter', 'exposure', 'target', 'site', 'width', 'binning']
    ]

    temps, airtemps, sunalts, moonalts, moondists, moonphases, ambtemps, imgids, biasavgs, dates, naxes1, naxes2 = [
        np.array([row['keywords'].get(key, np.nan) for row in res])
        for key in ['CCD_TEMP', 'CCD_AIR', 'SUN_ALT', 'MOONALT', 'MOONDIST', 'MOONPHA', 'AMBTEMP', 'IMGID', 'BIASAVG', 'DATE-OBS', 'NAXIS1', 'NAXIS2']
    ]

    biasavgs = biasavgs.astype(np.double)
    tasks = []

    for cfg in calibration_configs:
        idx = (serials == cfg['serial']) & (binnings == cfg['binning'])
        if 'date-before' in cfg:
            idx &= dates < cfg['date-before']
        if 'date-after' in cfg:
            idx &= dates > cfg['date-after']
        if 'width' in cfg:
            idx &= widths == cfg['width']

        idx1 = idx \
            & (temps < options.max_temp) \
            & (means < cfg.get('means_max', 1000)) \
            & (means > cfg.get('means_min', 0))

        idx1 &= ((targets == 21) & (sunalts < -18) & (moonphases > 20)) \
            | ((targets == 1) & (sunalts < -1) & (moonphases > 20)) \
            | ((targets == 2000) & (sunalts < -10) & (moonphases > 20)) \
            | ((targets == 2) & (sunalts < -6)) \
            | (targets == 20)

        if len(means[idx1]) < 10:
            continue

        if 'airtemp_a' in cfg:
            bias = airtemps * cfg['airtemp_a'] + cfg['airtemp_b']
        else:
            bias = np.zeros_like(means)

        bias[np.isfinite(biasavgs)] = biasavgs[np.isfinite(biasavgs)]

        for exp in np.unique(exposures[idx1]):
            eidx = exposures == exp
            mean = rmean((means - bias)[idx1 & eidx])
            std = rstd((means - bias)[idx1 & eidx])
            idx1[eidx] &= np.abs((means - bias)[eidx] - mean) < 3.0 * std

        for site in np.unique(sites[idx1]):
            site_idx = idx1 & (sites == site)
            for ccd in np.unique(ccds[site_idx]):
                ccd_idx = site_idx & (ccds == ccd)
                fsizes = np.unique(list(zip(naxes1[ccd_idx], naxes2[ccd_idx])), axis=0)

                for fsize in fsizes:
                    idx11 = idx1 & (sites == site) & (ccds == ccd) & (naxes1 == fsize[0]) & (naxes2 == fsize[1])
                    idx01 = idx & (sites == site) & (ccds == ccd) & (naxes1 == fsize[0]) & (naxes2 == fsize[1])

                    if not np.any(idx01):
                        continue

                    night1 = nights[idx01][0]
                    night2 = night1
                    last_night = nights[idx01][-1]

                    while True:
                        if night1 > last_night or night2 > last_night:
                            break

                        night2 = get_next_month(night2)
                        idx2 = idx11 & (nights >= night1) & (nights < night2)
                        counts = np.unique(exposures[idx2], return_counts=True)[1]

                        if np.sum(counts >= options.min_images) >= 4:
                            segment_night1 = night1
                            basename = os.path.join(
                                options.basedir,
                                site,
                                'masterdarks',
                                'dark_%s_%s_%s_%s_%s_%s' % (
                                    site,
                                    ccd,
                                    cfg['serial'],
                                    segment_night1,
                                    cfg['binning'],
                                    '%sx%s' % (fsize[0], fsize[1]),
                                ),
                            )
                            bias_name = basename + '_bias.fits'
                            dcurrent_name = basename + '_dcurrent.fits'
                            need_derived = options.replace or not os.path.exists(bias_name) or not os.path.exists(dcurrent_name)

                            exposure_groups = []
                            progress_total = 0
                            all_darks_exist = True
                            for exp in np.unique(exposures[idx2]):
                                idx3 = idx2 & (exposures == exp)
                                group_filenames = filenames[idx3].tolist()
                                dark_name = basename + '_%s.fits' % exp
                                dark_exists = os.path.exists(dark_name)

                                if not dark_exists:
                                    all_darks_exist = False

                                if len(group_filenames) < 6:
                                    group_progress = 0
                                elif dark_exists and not options.replace:
                                    group_progress = 1 if need_derived else 0
                                else:
                                    group_progress = len(group_filenames)

                                exposure_groups.append({
                                    'exposure': to_python_scalar(exp),
                                    'filenames': group_filenames,
                                    'progress_total': group_progress,
                                })
                                progress_total += group_progress

                            if all_darks_exist and not need_derived:
                                night1 = night2
                                continue

                            if need_derived:
                                progress_total += 2

                            tasks.append({
                                'basename': basename,
                                'cfg': cfg.copy(),
                                'replace': options.replace,
                                'show_progress': options.nthreads == 1,
                                'header_filename': filenames[idx2][0],
                                'exposure_groups': exposure_groups,
                                'n_selected': int(np.sum(idx2)),
                                'progress_total': progress_total,
                                'need_derived': need_derived,
                            })

                            night1 = night2

    return tasks


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog [options] arg")

    parser.add_option('-B', '--basedir', help='Base directory for output files', action='store', dest='basedir', type='str', default='calibrations')
    parser.add_option('-s', '--site', help='Site', action='store', dest='site', type='str', default=None)
    parser.add_option('-c', '--ccd', help='CCD', action='store', dest='ccd', type='str', default=None)
    parser.add_option('--serial', help='Camera serial number', action='store', dest='serial', type='int', default=None)
    parser.add_option('-t', '--target', help='Image target', action='store', dest='target', type='int', default=None)
    parser.add_option('-f', '--filter', help='Filter', action='store', dest='filter', type='str', default=None)
    parser.add_option('-b', '--binning', help='Binning', action='store', dest='binning', type='str', default=None)
    parser.add_option('-e', '--exposure', help='Exposure', action='store', dest='exposure', type='float', default=None)
    parser.add_option('-n', '--night', help='Night of observations', action='store', dest='night', type='str', default=None)
    parser.add_option('--night1', help='First night of observations', action='store', dest='night1', type='str', default=None)
    parser.add_option('--night2', help='Last night of observations', action='store', dest='night2', type='str', default=None)
    parser.add_option('-j', '--nthreads', help='Number of worker processes', action='store', dest='nthreads', type='int', default=1)

    parser.add_option('--max-temp', help='Maximal permitted temperature', action='store', dest='max_temp', type='float', default=-19)
    parser.add_option('--min-images', help='Minimal number of images per exposure', action='store', dest='min_images', type='int', default=10)

    parser.add_option('-r', '--replace', help='Replace existing files', action='store_true', dest='replace', default=False)

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

    if options.target is not None:
        print('Searching for images with target', options.target, file=sys.stderr)
        wheres += ['target=%s']
        wargs += [options.target]

    if options.filter is not None:
        print('Searching for images with filter', options.filter, file=sys.stderr)
        wheres += ['filter=%s']
        wargs += [options.filter]

    if options.binning is not None:
        print('Searching for images with binning', options.binning, file=sys.stderr)
        wheres += ['binning=%s']
        wargs += [options.binning]

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

    res = fram.query('SELECT * FROM images WHERE ' + ' AND '.join(wheres) + ' ORDER BY time ', wargs)
    print(len(res), 'dark images found', file=sys.stderr)

    if not len(res):
        sys.exit(0)

    tasks = build_tasks(res, options)
    print(len(tasks), 'segments to process using', options.nthreads, 'worker(s)', file=sys.stderr)

    if not len(tasks):
        sys.exit(0)

    ok = True

    if options.nthreads > 1:
        import multiprocessing

        total_progress = sum(task['progress_total'] for task in tasks)
        progress_queue = multiprocessing.Queue()
        progress_bar = tqdm(total=total_progress, desc='Masterdarks', unit='step')
        progress_thread = threading.Thread(
            target=progress_consumer,
            args=(progress_queue, progress_bar),
            daemon=True,
        )
        progress_thread.start()

        pool = multiprocessing.Pool(
            options.nthreads,
            initializer=init_worker,
            initargs=(progress_queue,),
        )

        try:
            for result in pool.imap_unordered(process_segment, tasks, 1):
                ok &= report_result(result)
        finally:
            pool.close()
            pool.join()
            progress_queue.put(None)
            progress_thread.join()
            progress_bar.close()

    else:
        for task in tasks:
            ok &= report_result(process_segment(task))

    if not ok:
        sys.exit(1)
