#!/usr/bin/env python3

import datetime
import os
import sys
import threading
import traceback

import numpy as np

from astropy.io import fits

from fram.calibrate import calibration_configs, crop_overscans
from fram.calibrate import rmean, rstd
from fram.fram import Fram
from stdpipe.utils import fits_write

try:
    from tqdm.auto import tqdm
except:
    from tqdm import tqdm

PROGRESS_QUEUE = None
STACK_BATCH_SIZE = 3
MIN_FRAMES_PER_MASTERDARK = 6
MIN_EXPOSURES_FOR_DERIVED_PRODUCTS = 2


def to_python_scalar(value):
    return value.item() if hasattr(value, 'item') else value


def get_next_month(night):
    timestamp = datetime.datetime.strptime(night, '%Y%m%d')
    year = timestamp.year
    month = timestamp.month + 1

    if month > 12:
        year += 1
        month = 1

    return datetime.datetime(year, month, 1).strftime('%Y%m%d')


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


def build_masterdark_header_template(header_filename, cfg):
    header = fits.getheader(header_filename, -1).copy()

    if 'airtemp_a' in cfg and 'airtemp_b' in cfg:
        header['AIRTEMPA'] = cfg['airtemp_a']
        header['AIRTEMPB'] = cfg['airtemp_b']

    if header.get('DATASEC'):
        header.pop('DATASEC')

    return header


def flush_dark_stack(sum_image, images):
    batch_size = len(images)
    median_image = np.median(images, axis=0)

    if sum_image is None:
        return batch_size * median_image, batch_size

    return sum_image + batch_size * median_image, batch_size


def build_masterdark(filenames, cfg, header_template, show_progress=False):
    stacked_sum = None
    pending_images = []
    progress_buffer = 0
    n_used = 0
    n_medians = 0
    header_out = header_template.copy()

    for filename in tqdm(filenames, leave=False, disable=not show_progress):
        image = fits.getdata(filename, -1).astype(np.double)
        header = fits.getheader(filename, -1)
        image, header = crop_overscans(image, header, cfg=cfg)

        if header.get('DATASEC0'):
            header_out['DATASEC0'] = header.get('DATASEC0')

        pending_images.append(image)
        progress_buffer += 1

        if progress_buffer >= 10:
            emit_progress(progress_buffer)
            progress_buffer = 0

        if len(pending_images) == STACK_BATCH_SIZE:
            stacked_sum, batch_size = flush_dark_stack(stacked_sum, pending_images)
            pending_images = []
            n_used += batch_size
            n_medians += 1

    if pending_images:
        stacked_sum, batch_size = flush_dark_stack(stacked_sum, pending_images)
        n_used += batch_size
        n_medians += 1

    if progress_buffer:
        emit_progress(progress_buffer)

    if not n_used:
        raise RuntimeError('No dark frames were stacked')

    masterdark = stacked_sum / n_used
    header_out['NDARKS'] = n_used
    header_out['NDARKMED'] = n_medians

    return masterdark, header_out


def fit_bias_and_dcurrent_maps(darks):
    sorted_exposures = sorted(darks)
    exposure_values = np.array(sorted_exposures, dtype=np.double)
    stacked_darks = np.array([darks[exposure]['dark'].ravel() for exposure in sorted_exposures])
    slope, intercept = np.polyfit(exposure_values, stacked_darks, 1)

    shape = darks[sorted_exposures[0]]['dark'].shape
    bias = intercept.reshape(shape)
    dcurrent = slope.reshape(shape)

    header = darks[sorted_exposures[0]]['header'].copy()
    header['EXPOSURE'] = 0

    return bias, dcurrent, header


def load_existing_masterdark(dark_name, header_template):
    dark = fits.getdata(dark_name, -1).astype(np.double)
    existing_header = fits.getheader(dark_name, -1)
    header = header_template.copy()

    if existing_header.get('DATASEC0'):
        header['DATASEC0'] = existing_header.get('DATASEC0')

    return dark, header


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
    need_derived_products = task['need_derived']

    os.makedirs(os.path.dirname(basename), exist_ok=True)

    header_template = build_masterdark_header_template(task['header_filename'], task['cfg'])
    darks = {}
    sparse_exposures = []

    for exposure_group in task['exposure_groups']:
        exposure = exposure_group['exposure']
        filenames = exposure_group['filenames']
        dark_name = basename + '_%s.fits' % exposure
        should_write_dark = True

        if len(filenames) < MIN_FRAMES_PER_MASTERDARK:
            sparse_exposures.append((exposure, len(filenames)))
            continue

        if os.path.exists(dark_name) and not task['replace']:
            if not need_derived_products:
                continue

            dark, header = load_existing_masterdark(dark_name, header_template)
            emit_progress(exposure_group['progress_total'])
            should_write_dark = False
        else:
            dark, header = build_masterdark(
                filenames,
                task['cfg'],
                header_template,
                show_progress=task['show_progress'],
            )

        header = header.copy()
        header['EXPOSURE'] = exposure
        header['IMAGETYP'] = 'masterdark'

        # Remove keywords that are breaking floating point images
        for kw in ['BZERO', 'BSCALE']:
            if kw in header:
                header.pop(kw)

        if should_write_dark:
            fits_write(dark_name, dark, header, compress=True)

        darks[exposure] = {
            'dark': dark,
            'header': header,
        }

    wrote_bias = False
    wrote_dcurrent = False

    if need_derived_products and len(darks) >= MIN_EXPOSURES_FOR_DERIVED_PRODUCTS:
        bias, dcurrent, header = fit_bias_and_dcurrent_maps(darks)

        header['IMAGETYP'] = 'bias'
        fits_write(bias_name, bias, header, compress=True)
        emit_progress(1)
        wrote_bias = True

        header['IMAGETYP'] = 'dcurrent'
        fits_write(dcurrent_name, dcurrent, header, compress=True)
        emit_progress(1)
        wrote_dcurrent = True

    return {
        'status': 'ok',
        'basename': basename,
        'n_selected': task['n_selected'],
        'n_darks': len(darks),
        'sparse_exposures': sparse_exposures,
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

    if result['sparse_exposures']:
        msg += ' [skipped %d sparse exposures]' % len(result['sparse_exposures'])

    tqdm.write(msg)
    return True


def load_dark_arrays(rows):
    return {
        'mean': np.array([row['mean'] for row in rows]),
        'median': np.array([row['median'] for row in rows]),
        'ccd': np.array([row['ccd'] for row in rows]),
        'serial': np.array([row['serial'] for row in rows]),
        'night': np.array([row['night'] for row in rows]),
        'filename': np.array([row['filename'] for row in rows]),
        'filter': np.array([row['filter'] for row in rows]),
        'exposure': np.array([row['exposure'] for row in rows]),
        'target': np.array([row['target'] for row in rows]),
        'site': np.array([row['site'] for row in rows]),
        'width': np.array([row['width'] for row in rows]),
        'binning': np.array([row['binning'] for row in rows]),
    }


def load_dark_keyword_arrays(rows):
    return {
        'CCD_TEMP': np.array([row['keywords'].get('CCD_TEMP', np.nan) for row in rows]),
        'CCD_AIR': np.array([row['keywords'].get('CCD_AIR', np.nan) for row in rows]),
        'SUN_ALT': np.array([row['keywords'].get('SUN_ALT', np.nan) for row in rows]),
        'MOONALT': np.array([row['keywords'].get('MOONALT', np.nan) for row in rows]),
        'MOONDIST': np.array([row['keywords'].get('MOONDIST', np.nan) for row in rows]),
        'MOONPHA': np.array([row['keywords'].get('MOONPHA', np.nan) for row in rows]),
        'AMBTEMP': np.array([row['keywords'].get('AMBTEMP', np.nan) for row in rows]),
        'IMGID': np.array([row['keywords'].get('IMGID', np.nan) for row in rows]),
        'BIASAVG': np.array([row['keywords'].get('BIASAVG', np.nan) for row in rows]).astype(np.double),
        'DATE-OBS': np.array([row['keywords'].get('DATE-OBS', np.nan) for row in rows]),
        'NAXIS1': np.array([row['keywords'].get('NAXIS1', np.nan) for row in rows]),
        'NAXIS2': np.array([row['keywords'].get('NAXIS2', np.nan) for row in rows]),
    }


def get_config_mask(cfg, arrays, keyword_arrays):
    config_mask = (arrays['serial'] == cfg['serial']) & (arrays['binning'] == cfg['binning'])

    if 'date-before' in cfg:
        config_mask &= keyword_arrays['DATE-OBS'] < cfg['date-before']
    if 'date-after' in cfg:
        config_mask &= keyword_arrays['DATE-OBS'] > cfg['date-after']
    if 'width' in cfg:
        config_mask &= arrays['width'] == cfg['width']

    return config_mask


def get_target_selection_mask(arrays, keyword_arrays):
    targets = arrays['target']
    sun_alts = keyword_arrays['SUN_ALT']
    moon_phases = keyword_arrays['MOONPHA']

    return (
        ((targets == 21) & (sun_alts < -18) & (moon_phases > 20))
        | ((targets == 1) & (sun_alts < -1) & (moon_phases > 20))
        | ((targets == 2000) & (sun_alts < -10) & (moon_phases > 20))
        | ((targets == 2) & (sun_alts < -6))
        | (targets == 20)
    )


def estimate_bias_levels(cfg, arrays, keyword_arrays):
    if 'airtemp_a' in cfg:
        bias_levels = keyword_arrays['CCD_AIR'] * cfg['airtemp_a'] + cfg['airtemp_b']
    else:
        bias_levels = np.zeros_like(arrays['mean'])

    finite_bias = np.isfinite(keyword_arrays['BIASAVG'])
    bias_levels[finite_bias] = keyword_arrays['BIASAVG'][finite_bias]

    return bias_levels


def reject_exposure_outliers(selected_mask, arrays, bias_levels):
    cleaned_mask = selected_mask.copy()
    means_minus_bias = arrays['mean'] - bias_levels

    for exposure in np.unique(arrays['exposure'][selected_mask]):
        exposure_mask = cleaned_mask & (arrays['exposure'] == exposure)
        if not np.any(exposure_mask):
            continue

        exposure_values = means_minus_bias[exposure_mask]
        mean_value = rmean(exposure_values)
        std_value = rstd(exposure_values)

        if not np.isfinite(std_value) or std_value == 0:
            continue

        cleaned_mask[exposure_mask] &= np.abs(exposure_values - mean_value) < 3.0 * std_value

    return cleaned_mask


def iter_frame_sizes(mask, keyword_arrays):
    widths = keyword_arrays['NAXIS1'][mask]
    heights = keyword_arrays['NAXIS2'][mask]

    if not len(widths):
        return []

    return np.unique(np.column_stack((widths, heights)), axis=0)


def build_segment_basename(basedir, site, ccd, cfg, segment_start_night, frame_size):
    width, height = frame_size
    return os.path.join(
        basedir,
        site,
        'masterdarks',
        'dark_%s_%s_%s_%s_%s_%sx%s' % (
            site,
            ccd,
            cfg['serial'],
            segment_start_night,
            cfg['binning'],
            width,
            height,
        ),
    )


def get_exposure_group_progress(filenames, dark_exists, replace, need_derived_products):
    if len(filenames) < MIN_FRAMES_PER_MASTERDARK:
        return 0

    if dark_exists and not replace:
        return 1 if need_derived_products else 0

    return len(filenames)


def build_exposure_groups(shard_mask, arrays, basename, options, need_derived_products):
    exposure_groups = []
    progress_total = 0
    all_masterdarks_exist = True

    for exposure in np.unique(arrays['exposure'][shard_mask]):
        exposure_mask = shard_mask & (arrays['exposure'] == exposure)
        group_filenames = arrays['filename'][exposure_mask].tolist()
        dark_name = basename + '_%s.fits' % exposure
        dark_exists = os.path.exists(dark_name)

        if not dark_exists:
            all_masterdarks_exist = False

        group_progress = get_exposure_group_progress(
            group_filenames,
            dark_exists,
            options.replace,
            need_derived_products,
        )

        exposure_groups.append({
            'exposure': to_python_scalar(exposure),
            'filenames': group_filenames,
            'progress_total': group_progress,
        })
        progress_total += group_progress

    return exposure_groups, progress_total, all_masterdarks_exist


def build_segment_task(cfg, site, ccd, frame_size, segment_start_night, shard_mask, arrays, options):
    basename = build_segment_basename(
        options.basedir,
        site,
        ccd,
        cfg,
        segment_start_night,
        frame_size,
    )

    need_derived_products = (
        options.replace
        or not os.path.exists(basename + '_bias.fits')
        or not os.path.exists(basename + '_dcurrent.fits')
    )

    exposure_groups, progress_total, all_masterdarks_exist = build_exposure_groups(
        shard_mask,
        arrays,
        basename,
        options,
        need_derived_products,
    )

    if all_masterdarks_exist and not need_derived_products:
        return None

    if need_derived_products:
        progress_total += 2

    return {
        'basename': basename,
        'cfg': cfg.copy(),
        'replace': options.replace,
        'show_progress': options.nthreads == 1,
        'header_filename': arrays['filename'][shard_mask][0],
        'exposure_groups': exposure_groups,
        'n_selected': int(np.sum(shard_mask)),
        'progress_total': progress_total,
        'need_derived': need_derived_products,
    }


def build_tasks(rows, options):
    rows.sort('time')

    arrays = load_dark_arrays(rows)
    keyword_arrays = load_dark_keyword_arrays(rows)
    tasks = []

    for cfg in calibration_configs:
        config_mask = get_config_mask(cfg, arrays, keyword_arrays)
        if not np.any(config_mask):
            continue

        selected_mask = (
            config_mask
            & (keyword_arrays['CCD_TEMP'] < options.max_temp)
            & (arrays['mean'] < cfg.get('means_max', 1000))
            & (arrays['mean'] > cfg.get('means_min', 0))
        )
        selected_mask &= get_target_selection_mask(arrays, keyword_arrays)

        if np.sum(selected_mask) < 10:
            continue

        bias_levels = estimate_bias_levels(cfg, arrays, keyword_arrays)
        selected_mask = reject_exposure_outliers(selected_mask, arrays, bias_levels)

        for site in np.unique(arrays['site'][selected_mask]):
            site_mask = selected_mask & (arrays['site'] == site)

            for ccd in np.unique(arrays['ccd'][site_mask]):
                camera_mask = site_mask & (arrays['ccd'] == ccd)

                for frame_size in iter_frame_sizes(camera_mask, keyword_arrays):
                    width, height = frame_size
                    selected_group_mask = (
                        selected_mask
                        & (arrays['site'] == site)
                        & (arrays['ccd'] == ccd)
                        & (keyword_arrays['NAXIS1'] == width)
                        & (keyword_arrays['NAXIS2'] == height)
                    )
                    all_group_mask = (
                        config_mask
                        & (arrays['site'] == site)
                        & (arrays['ccd'] == ccd)
                        & (keyword_arrays['NAXIS1'] == width)
                        & (keyword_arrays['NAXIS2'] == height)
                    )

                    if not np.any(all_group_mask):
                        continue

                    segment_start_night = arrays['night'][all_group_mask][0]
                    segment_end_night = segment_start_night
                    last_night = arrays['night'][all_group_mask][-1]

                    while True:
                        if segment_start_night > last_night or segment_end_night > last_night:
                            break

                        segment_end_night = get_next_month(segment_end_night)
                        shard_mask = (
                            selected_group_mask
                            & (arrays['night'] >= segment_start_night)
                            & (arrays['night'] < segment_end_night)
                        )
                        counts = np.unique(arrays['exposure'][shard_mask], return_counts=True)[1]

                        if np.sum(counts >= options.min_images) < 4:
                            continue

                        task = build_segment_task(
                            cfg,
                            site,
                            ccd,
                            frame_size,
                            segment_start_night,
                            shard_mask,
                            arrays,
                            options,
                        )
                        if task is not None:
                            tasks.append(task)

                        segment_start_night = segment_end_night

    return tasks


def append_query_filter(where, args, value, sql_condition, description):
    if value is None:
        return

    print(description, value, file=sys.stderr)
    where.append(sql_condition)
    args.append(value)


def query_dark_rows(fram, options):
    where = ["(type='dark' or type='zero')"]
    args = []

    append_query_filter(where, args, options.site, 'site=%s', 'Searching for images from site')
    append_query_filter(where, args, options.ccd, 'ccd=%s', 'Searching for images from ccd')
    append_query_filter(where, args, options.serial, 'serial=%s', 'Searching for images with serial')
    append_query_filter(where, args, options.target, 'target=%s', 'Searching for images with target')
    append_query_filter(where, args, options.filter, 'filter=%s', 'Searching for images with filter')
    append_query_filter(where, args, options.binning, 'binning=%s', 'Searching for images with binning')
    append_query_filter(where, args, options.exposure, 'exposure=%s', 'Searching for images with exposure')
    append_query_filter(where, args, options.night, 'night=%s', 'Searching for images from night')
    append_query_filter(where, args, options.night1, 'night>=%s', 'Searching for images night >=')
    append_query_filter(where, args, options.night2, 'night<=%s', 'Searching for images night <=')

    return fram.query('SELECT * FROM images WHERE ' + ' AND '.join(where) + ' ORDER BY time', args)


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
            for result in pool.imap_unordered(process_segment, tasks, 1):
                ok &= report_result(result)
        finally:
            pool.close()
            pool.join()
            progress_queue.put(None)
            progress_thread.join()
            progress_bar.close()

        return ok

    for task in tasks:
        ok &= report_result(process_segment(task))

    return ok


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage='usage: %prog [options] arg')

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

    fram = Fram(dbname=options.db, dbhost=options.dbhost)
    if not fram:
        print("Can't connect to the database", file=sys.stderr)
        sys.exit(1)

    dark_rows = query_dark_rows(fram, options)
    print(len(dark_rows), 'dark images found', file=sys.stderr)
    if not len(dark_rows):
        sys.exit(0)

    tasks = build_tasks(dark_rows, options)
    print(len(tasks), 'segments to process using', options.nthreads, 'worker(s)', file=sys.stderr)
    if not len(tasks):
        sys.exit(0)

    if not run_tasks(tasks, options.nthreads, 'Masterdarks'):
        sys.exit(1)
