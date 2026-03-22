#!/usr/bin/env python3

import bisect
import os
import sys
import threading
import traceback

import numpy as np

from astropy.io import fits

from fram.calibrate import calibrate
from fram.fram import Fram
from stdpipe.utils import fits_write

try:
    from tqdm.auto import tqdm
except:
    from tqdm import tqdm

PROGRESS_QUEUE = None
IGNORED_CCDS = {'WF5'}
IGNORED_FILTERS = {'D', 'DF'}


def to_python_scalar(value):
    return value.item() if hasattr(value, 'item') else value


def normalize_exposure(exposure):
    if exposure is None:
        return None
    return float(to_python_scalar(exposure))


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


def add_index_entry(index, key, night, filename):
    index.setdefault(key, []).append((night, filename))


def finalize_index(index):
    for key, entries in list(index.items()):
        entries.sort()
        index[key] = {
            'nights': [night for night, _ in entries],
            'filenames': [filename for _, filename in entries],
        }


def build_calibration_indexes(rows):
    masterdark_index = {}
    bias_index = {}
    dcurrent_index = {}

    for row in rows:
        ccd = row.get('ccd')
        serial = row.get('serial')
        width = row.get('cropped_width')
        height = row.get('cropped_height')

        if ccd is None or serial is None or width is None or height is None:
            continue

        key = (ccd, int(serial), int(width), int(height))
        night = row['night']
        filename = row['filename']
        image_type = row['type']

        if image_type == 'masterdark':
            exposure = normalize_exposure(row.get('exposure'))
            if exposure is None:
                continue
            add_index_entry(masterdark_index, key + (exposure,), night, filename)
        elif image_type == 'bias':
            add_index_entry(bias_index, key, night, filename)
        elif image_type == 'dcurrent':
            add_index_entry(dcurrent_index, key, night, filename)

    finalize_index(masterdark_index)
    finalize_index(bias_index)
    finalize_index(dcurrent_index)

    return {
        'masterdark': masterdark_index,
        'bias': bias_index,
        'dcurrent': dcurrent_index,
    }


def find_nearest_calibration(index, key, night):
    matches = index.get(key)
    if not matches:
        return None

    pos = bisect.bisect_right(matches['nights'], night)
    if pos:
        return matches['filenames'][pos - 1]
    return matches['filenames'][0]


def find_dark_source(indexes, night, ccd, serial, width, height, exposure):
    key = (ccd, int(serial), int(width), int(height))
    exposure = normalize_exposure(exposure)

    masterdark_name = find_nearest_calibration(indexes['masterdark'], key + (exposure,), night)
    if masterdark_name is not None:
        return {'type': 'masterdark', 'filename': masterdark_name}

    bias_name = find_nearest_calibration(indexes['bias'], key, night)
    dcurrent_name = find_nearest_calibration(indexes['dcurrent'], key, night)
    if bias_name is not None and dcurrent_name is not None:
        return {'type': 'model', 'bias_name': bias_name, 'dcurrent_name': dcurrent_name}

    return None


def get_group_key(row):
    site = row.get('site')
    night = row.get('night')
    ccd = row.get('ccd')
    filter_name = row.get('filter')
    binning = row.get('binning')
    serial = row.get('serial')
    width = row.get('cropped_width')
    height = row.get('cropped_height')

    if site is None or night is None or ccd is None or filter_name is None:
        return None
    if binning is None or serial is None or width is None or height is None:
        return None
    if ccd in IGNORED_CCDS or filter_name in IGNORED_FILTERS:
        return None

    return (
        site,
        night,
        ccd,
        filter_name,
        binning,
        int(width),
        int(height),
        int(serial),
    )


def build_nightly_flat_basename(basedir, site, night, ccd, serial, binning, width, height, filter_name):
    return os.path.join(
        basedir,
        site,
        'flats',
        'flat_%s_%s_%s_%s_%s_%sx%s_%s' % (
            site,
            ccd,
            serial,
            night,
            binning,
            width,
            height,
            filter_name,
        ),
    )


def group_flat_rows(rows):
    grouped_rows = {}

    for row in rows:
        group_key = get_group_key(row)
        if group_key is None:
            continue

        grouped_rows.setdefault(group_key, []).append({
            'filename': row['filename'],
            'exposure': normalize_exposure(row.get('exposure')),
        })

    return grouped_rows


def select_dark_sources(group_rows, calibration_indexes, night, ccd, serial, width, height):
    usable_sources = {}

    unique_exposures = sorted({
        row['exposure']
        for row in group_rows
        if row['exposure'] is not None
    })

    for exposure in unique_exposures:
        source = find_dark_source(
            calibration_indexes,
            night,
            ccd,
            serial,
            width,
            height,
            exposure,
        )
        if source is not None:
            usable_sources[exposure] = source

    return usable_sources


def build_nightly_flat_task(group_key, group_rows, calibration_indexes, options):
    site, night, ccd, filter_name, binning, width, height, serial = group_key
    sorted_rows = sorted(group_rows, key=lambda row: row['filename'])

    if len(sorted_rows) < options.min_images:
        return None

    basename = build_nightly_flat_basename(
        options.basedir,
        site,
        night,
        ccd,
        serial,
        binning,
        width,
        height,
        filter_name,
    )

    if os.path.exists(basename + '_min.fits') and not options.replace:
        return None

    usable_sources = select_dark_sources(
        sorted_rows,
        calibration_indexes,
        night,
        ccd,
        serial,
        width,
        height,
    )

    usable_rows = [row for row in sorted_rows if row['exposure'] in usable_sources]
    if not usable_rows:
        return None

    task_sources = []
    calibration_filenames = set()

    for exposure in sorted(usable_sources):
        source = usable_sources[exposure].copy()
        source['exposure'] = exposure
        task_sources.append(source)

        if source['type'] == 'masterdark':
            calibration_filenames.add(source['filename'])
        else:
            calibration_filenames.add(source['bias_name'])
            calibration_filenames.add(source['dcurrent_name'])

    return {
        'basename': basename,
        'files': usable_rows,
        'dark_sources': task_sources,
        'n_input': len(sorted_rows),
        'n_selected': len(usable_rows),
        'progress_total': len(calibration_filenames) + len(usable_rows) + 2,
        'show_progress': options.nthreads == 1,
    }


def build_tasks(rows, calibration_indexes, options):
    grouped_rows = group_flat_rows(rows)
    tasks = []

    for group_key in sorted(grouped_rows):
        task = build_nightly_flat_task(
            group_key,
            grouped_rows[group_key],
            calibration_indexes,
            options,
        )
        if task is not None:
            tasks.append(task)

    return tasks


def load_dark_map(task):
    dark_frames = {}
    loaded_frames = {}
    n_masterdarks = 0
    n_modeled = 0

    for source in task['dark_sources']:
        exposure = source['exposure']

        if source['type'] == 'masterdark':
            dark_frames[exposure] = fits.getdata(source['filename'], -1).astype(np.double)
            emit_progress(1)
            n_masterdarks += 1
            continue

        bias_name = source['bias_name']
        dcurrent_name = source['dcurrent_name']

        if bias_name not in loaded_frames:
            loaded_frames[bias_name] = fits.getdata(bias_name, -1).astype(np.double)
            emit_progress(1)

        if dcurrent_name not in loaded_frames:
            loaded_frames[dcurrent_name] = fits.getdata(dcurrent_name, -1).astype(np.double)
            emit_progress(1)

        dark_frames[exposure] = loaded_frames[bias_name] + exposure * loaded_frames[dcurrent_name]
        n_modeled += 1

    return dark_frames, n_masterdarks, n_modeled


def build_nightly_products(files, dark_frames, show_progress=False):
    min_image = None
    mean_image = None
    m2_image = None
    header_out = None
    nused = 0

    for file_info in tqdm(files, leave=False, disable=not show_progress):
        filename = file_info['filename']
        exposure = file_info['exposure']
        dark = dark_frames.get(exposure)

        if dark is None:
            continue

        image = fits.getdata(filename, -1).astype(np.double)
        header = fits.getheader(filename, -1)
        image, header = calibrate(image, header, dark=dark)
        emit_progress(1)

        if not np.isfinite(np.min(image)):
            raise RuntimeError('Non-finite calibrated image: %s' % filename)

        median = np.median(image)
        if not np.isfinite(median) or median == 0:
            raise RuntimeError('Invalid normalization median for %s' % filename)

        normalized_image = image / median
        if not np.isfinite(np.min(normalized_image)):
            raise RuntimeError('Non-finite normalized image: %s' % filename)

        nused += 1

        if header_out is None:
            header_out = header.copy()

        if min_image is None:
            min_image = normalized_image.copy()
            mean_image = normalized_image.copy()
            m2_image = np.zeros_like(normalized_image)
            continue

        min_image = np.minimum(min_image, normalized_image)

        delta = normalized_image - mean_image
        mean_image += delta / nused
        delta2 = normalized_image - mean_image
        m2_image += delta * delta2

    if not nused:
        raise RuntimeError('No flat frames were calibrated')

    std_image = np.sqrt(m2_image / nused)
    return min_image, std_image, header_out, nused


def write_nightly_products(basename, min_image, std_image, header_out):
    min_name = basename + '_min.fits'
    std_name = basename + '_std.fits'

    fits_write(min_name, min_image, header_out, compress=True)
    emit_progress(1)

    std_header = header_out.copy()
    std_header['STDMEAN'] = float(np.mean(std_image))
    std_header['STDRMS'] = float(np.std(std_image))
    std_header['STDMED'] = float(np.median(std_image))
    fits_write(std_name, std_image, std_header, compress=True)
    emit_progress(1)

    return True, True


def process_task(task):
    try:
        return _process_task(task)
    except KeyboardInterrupt:
        raise
    except:
        return {
            'status': 'error',
            'basename': task['basename'],
            'traceback': traceback.format_exc(),
        }


def _process_task(task):
    basename = task['basename']
    os.makedirs(os.path.dirname(basename), exist_ok=True)

    dark_frames, n_masterdarks, n_modeled = load_dark_map(task)
    min_image, std_image, header_out, nused = build_nightly_products(
        task['files'],
        dark_frames,
        show_progress=task['show_progress'],
    )
    wrote_min, wrote_std = write_nightly_products(
        basename,
        min_image,
        std_image,
        header_out,
    )

    return {
        'status': 'ok',
        'basename': basename,
        'n_input': task['n_input'],
        'n_selected': task['n_selected'],
        'n_used': nused,
        'n_exposures': len(task['dark_sources']),
        'n_masterdarks': n_masterdarks,
        'n_modeled': n_modeled,
        'wrote_min': wrote_min,
        'wrote_std': wrote_std,
        'skipped_files': task['n_input'] - task['n_selected'],
    }


def report_result(result):
    if result['status'] == 'error':
        tqdm.write('ERROR while processing %s' % result['basename'], file=sys.stderr)
        tqdm.write(result['traceback'], file=sys.stderr)
        return False

    msg = '%s : %d calibrated exposure groups, %d usable files (%d total)' % (
        result['basename'],
        result['n_exposures'],
        result['n_used'],
        result['n_input'],
    )

    details = []
    if result['n_masterdarks']:
        details.append('%d masterdarks' % result['n_masterdarks'])
    if result['n_modeled']:
        details.append('%d modeled darks' % result['n_modeled'])
    if result['skipped_files']:
        details.append('%d files without darks' % result['skipped_files'])
    if result['wrote_min'] or result['wrote_std']:
        outputs = []
        if result['wrote_min']:
            outputs.append('min')
        if result['wrote_std']:
            outputs.append('std')
        details.append('wrote %s' % '/'.join(outputs))

    if details:
        msg += ' [' + '; '.join(details) + ']'

    tqdm.write(msg)
    return True


def append_query_filter(where, args, value, sql_condition, description):
    if value is None:
        return

    print(description, value, file=sys.stderr)
    where.append(sql_condition)
    args.append(value)


def query_flat_rows(fram, options):
    where = ["type='flat'"]
    args = []

    append_query_filter(where, args, options.site, 'site=%s', 'Searching for images from site')
    append_query_filter(where, args, options.ccd, 'ccd=%s', 'Searching for images from ccd')
    append_query_filter(where, args, options.serial, 'serial=%s', 'Searching for images with serial')
    append_query_filter(where, args, options.filter, 'filter=%s', 'Searching for images with filter')
    append_query_filter(where, args, options.binning, 'binning=%s', 'Searching for images with binning')
    append_query_filter(where, args, options.exposure, 'exposure=%s', 'Searching for images with exposure')
    append_query_filter(where, args, options.night, 'night=%s', 'Searching for images from night')
    append_query_filter(where, args, options.night1, 'night>=%s', 'Searching for images night >=')
    append_query_filter(where, args, options.night2, 'night<=%s', 'Searching for images night <=')

    return fram.query('SELECT * FROM images WHERE ' + ' AND '.join(where) + ' ORDER BY time', args)


def query_dark_calibration_rows(fram, options):
    where = ["(type='masterdark' or type='bias' or type='dcurrent')"]
    args = []

    if options.ccd is not None:
        where.append('ccd=%s')
        args.append(options.ccd)

    if options.serial is not None:
        where.append('serial=%s')
        args.append(options.serial)

    return fram.query(
        'SELECT filename,night,type,exposure,ccd,serial,cropped_width,cropped_height FROM calibrations WHERE ' + ' AND '.join(where),
        args,
    )


def run_tasks(tasks, nthreads, progress_label):
    ok = True

    if nthreads > 1:
        import multiprocessing

        total_progress = sum(task['progress_total'] for task in tasks)
        progress_queue = multiprocessing.Queue()
        progress_bar = tqdm(total=total_progress, desc=progress_label, unit='step')
        progress_thread = threading.Thread(
            target=progress_consumer,
            args=(progress_queue, progress_bar),
            daemon=True,
        )
        progress_thread.start()

        pool = multiprocessing.Pool(
            nthreads,
            initializer=init_worker,
            initargs=(progress_queue,),
        )

        try:
            for result in pool.imap_unordered(process_task, tasks, 1):
                ok &= report_result(result)
        finally:
            pool.close()
            pool.join()
            progress_queue.put(None)
            progress_thread.join()
            progress_bar.close()

        return ok

    for task in tasks:
        ok &= report_result(process_task(task))

    return ok


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage='usage: %prog [options] arg')

    parser.add_option('-B', '--basedir', help='Base directory for output files', action='store', dest='basedir', type='str', default='calibrations')
    parser.add_option('-s', '--site', help='Site', action='store', dest='site', type='str', default=None)
    parser.add_option('-c', '--ccd', help='CCD', action='store', dest='ccd', type='str', default=None)
    parser.add_option('--serial', help='Camera serial number', action='store', dest='serial', type='int', default=None)
    parser.add_option('-f', '--filter', help='Filter', action='store', dest='filter', type='str', default=None)
    parser.add_option('-b', '--binning', help='Binning', action='store', dest='binning', type='str', default=None)
    parser.add_option('-e', '--exposure', help='Exposure', action='store', dest='exposure', type='float', default=None)
    parser.add_option('-n', '--night', help='Night of observations', action='store', dest='night', type='str', default=None)
    parser.add_option('--night1', help='First night of observations', action='store', dest='night1', type='str', default=None)
    parser.add_option('--night2', help='Last night of observations', action='store', dest='night2', type='str', default=None)
    parser.add_option('-j', '--nthreads', help='Number of worker processes', action='store', dest='nthreads', type='int', default=1)

    parser.add_option('--min-images', help='Minimal number of flat frames per nightly group', action='store', dest='min_images', type='int', default=3)
    parser.add_option('-r', '--replace', help='Replace existing files', action='store_true', dest='replace', default=False)

    parser.add_option('-d', '--db', help='Database name', action='store', dest='db', type='str', default='fram')
    parser.add_option('-H', '--host', help='Database host', action='store', dest='dbhost', type='str', default=None)

    (options, args) = parser.parse_args()

    fram = Fram(dbname=options.db, dbhost=options.dbhost)
    if not fram:
        print("Can't connect to the database", file=sys.stderr)
        sys.exit(1)

    flat_rows = query_flat_rows(fram, options)
    print(len(flat_rows), 'flat images found', file=sys.stderr)
    if not len(flat_rows):
        sys.exit(0)

    calibration_rows = query_dark_calibration_rows(fram, options)
    print(len(calibration_rows), 'dark calibration files found', file=sys.stderr)

    calibration_indexes = build_calibration_indexes(calibration_rows)
    tasks = build_tasks(flat_rows, calibration_indexes, options)
    print(len(tasks), 'nightly flat tasks to process using', options.nthreads, 'worker(s)', file=sys.stderr)
    if not len(tasks):
        sys.exit(0)

    if not run_tasks(tasks, options.nthreads, 'Nightly flats'):
        sys.exit(1)
