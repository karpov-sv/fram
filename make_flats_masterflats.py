#!/usr/bin/env python3

import datetime
import glob
import os
import sys

import numpy as np

from astropy.io import fits

from fram.calibrate import rmean, rstd
from fram.fram import get_iso_time, get_night_time
from stdpipe.utils import fits_write

try:
    from tqdm.auto import tqdm
except:
    from tqdm import tqdm


def get_next_month(night):
    timestamp = datetime.datetime.strptime(night, '%Y%m%d')
    year = timestamp.year
    month = timestamp.month + 1

    if month > 12:
        year += 1
        month = 1

    return datetime.datetime(year, month, 1).strftime('%Y%m%d')


flat_configs = [
    # La Palma
    {'site':'cta-n', 'ccd':'C0', 'serial':2596, 'size':'1056x1024', 'last':'20190618'},
    # ...and then C0 got some dust due to opening by Dusan
    {'site':'cta-n', 'ccd':'C0', 'serial':2596, 'size':'1056x1024', 'first':'20190619', 'mode':'every'},
    # La Palma widefield
    {'site':'cta-n', 'ccd':'WF0', 'serial':6149, 'size':'4096x4096', 'first':'20181023', 'last':'20190909'},
    # After WF power repair
    {'site':'cta-n', 'ccd':'WF0', 'serial':6149, 'size':'4096x4096', 'first':'20191101', 'last':'20210222'},
    # After WF replacement on 2021-02-25
    {'site':'cta-n', 'ccd':'WF0', 'serial':6132, 'size':'4096x4096', 'first':'20210225', 'last':'20210923'},
    # After WF replacement on 2021-09-28
    {'site':'cta-n', 'ccd':'WF0', 'serial':6149, 'size':'4096x4096', 'first':'20210928', 'last':'20250302'},
    # After snowpile incident
    {'site':'cta-n', 'ccd':'WF0', 'serial':6029, 'size':'4096x4096', 'first':'20250416'},


    # Auger NF
    {'site':'auger', 'ccd':'NF3', 'serial':2328, 'binning':'1x1', 'size':'1536x1024', 'last1':'20181223'},

    {'site':'auger', 'ccd':'NF4', 'serial':6205, 'binning':'1x1', 'size':'4096x4096', 'last':'20181223'},
    {'site':'auger', 'ccd':'NF4', 'serial':6205, 'binning':'2x2', 'size':'2048x2048', 'last':'20181223'},

    {'site':'auger', 'ccd':'NF4', 'serial':6205, 'binning':'1x1', 'size':'4096x4096', 'first':'20190514', 'last':'20210213'},
    {'site':'auger', 'ccd':'NF4', 'serial':6205, 'binning':'2x2', 'size':'2048x2048', 'first':'20190514', 'last':'20210213'},

    {'site':'auger', 'ccd':'NF4', 'serial':40032, 'binning':'1x1', 'size':'4096x4096', 'last':'20241110'},
    {'site':'auger', 'ccd':'NF4', 'serial':40032, 'binning':'2x2', 'size':'2048x2048', 'last':'20241110'},

    {'site':'auger', 'ccd':'NF4', 'serial':6205, 'binning':'1x1', 'size':'4096x4096', 'first':'20241111'},
    {'site':'auger', 'ccd':'NF4', 'serial':6205, 'binning':'2x2', 'size':'2048x2048', 'first':'20241111'},

    # Auger WF
    {'site':'auger', 'ccd':'WF4', 'serial':6029, 'size':'4096x4096', 'first':'20160601'}, # ???
    {'site':'auger', 'ccd':'WF6', 'serial':6132, 'size':'4096x4096', 'first':'20161121', 'last':'20170120'}, # ???
    {'site':'auger', 'ccd':'WF6', 'serial':6132, 'size':'4096x4096', 'first':'20170120', 'last':'20170501'}, # ???
    {'site':'auger', 'ccd':'WF6', 'serial':6132, 'size':'4096x4096', 'first':'20170501'}, # ???
    {'site':'auger', 'ccd':'WF7', 'serial':6029, 'size':'4096x4096'},

    {'site':'auger', 'ccd':'WF8', 'serial':6204, 'binning':'1x1', 'size':'4096x4096', 'first':'20180913', 'last':'20190531'},
    {'site':'auger', 'ccd':'WF8', 'serial':6204, 'binning':'2x2', 'size':'2048x2048', 'first':'20180913', 'last':'20190531'},

    {'site':'auger', 'ccd':'WF8', 'serial':6029, 'binning':'1x1', 'size':'4096x4096', 'first':'20191029', 'last':'20211127'},
    {'site':'auger', 'ccd':'WF8', 'serial':6029, 'binning':'2x2', 'size':'2048x2048', 'first':'20191029', 'last':'20211127'},

    {'site':'auger', 'ccd':'WF8', 'serial':6132, 'binning':'1x1', 'size':'4096x4096', 'first':'20211211', 'last':'20220321'},
    {'site':'auger', 'ccd':'WF8', 'serial':6132, 'binning':'2x2', 'size':'2048x2048', 'first':'20211211', 'last':'20220321'},

    {'site':'auger', 'ccd':'WF8', 'serial':6205, 'binning':'1x1', 'size':'4096x4096', 'first':'20220330', 'last':'20230203'},
    {'site':'auger', 'ccd':'WF8', 'serial':6205, 'binning':'2x2', 'size':'2048x2048', 'first':'20220330', 'last':'20230203'},

    {'site':'auger', 'ccd':'WF8', 'serial':6069, 'binning':'1x1', 'size':'4096x4096', 'first':'20230211'},
    {'site':'auger', 'ccd':'WF8', 'serial':6069, 'binning':'2x2', 'size':'2048x2048', 'first':'20230211'},

    # Auger2 - WF0
    {'site':'auger2', 'ccd':'WF0', 'serial':40033, 'binning':'1x1', 'size':'4096x4096', 'last':'20241110'},
    {'site':'auger2', 'ccd':'WF0', 'serial':40032, 'binning':'1x1', 'size':'4096x4096', 'first':'20241112'},

    # Auger2 - WF1
    {'site':'auger2', 'ccd':'WF1', 'serial':80011, 'binning':'1x1', 'size':'4096x4096', 'last':'20241110'},
    {'site':'auger2', 'ccd':'WF1', 'serial':80059, 'binning':'1x1', 'size':'4096x4096', 'first':'20241112'},

    # S1 in Prague
    {'site':'cta-s1', 'ccd':'WF0', 'serial':6069, 'size':'4096x4096'},
    # S1 on Paranal
    {'site':'cta-s1', 'ccd':'WF0', 'serial':6132, 'size':'4096x4096'},
    # S1 with new CCD / improved readout
    {'site':'cta-s1', 'ccd':'WF0', 'serial':6204, 'size':'4096x4096'},

    # S0 on Paranal
    {'site':'cta-s0', 'ccd':'WF0', 'serial':6069, 'size':'4096x4096', 'last':'20170919'},
    {'site':'cta-s0', 'ccd':'WF0', 'serial':6069, 'size':'4096x4096', 'first':'20170920', 'last':'20180701'},
    # S0 6069 had cooling failure and replaced with 6132
    {'site':'cta-s0', 'ccd':'WF1', 'serial':6132, 'size':'4096x4096', 'first':'20180715', 'last':'20181226'},
    # S0 after repairs
    {'site':'cta-s0', 'ccd':'WF0', 'serial':6069, 'size':'4096x4096', 'first':'20191212', 'last':'20220324'},
    # S0 with 6029
    {'site':'cta-s0', 'ccd':'WF0', 'serial':6029, 'size':'4096x4096', 'first':'20220325', 'last':'20230214'},
    # S0 with 6132 again
    {'site':'cta-s0', 'ccd':'WF0', 'serial':6132, 'size':'4096x4096', 'first':'20230220'},
]


def load_nightly_flat_metadata(pattern):
    filenames = np.array(sorted(glob.glob(pattern)))
    if not len(filenames):
        return None

    sites = []
    ccds = []
    serials = []
    nights = []
    binnings = []
    sizes = []
    filters = []

    for filename in filenames:
        parts = os.path.split(filename)[-1].split('_')
        sites.append(parts[1])
        ccds.append(parts[2])
        serials.append(int(parts[3]))
        nights.append(parts[4])
        binnings.append(parts[5])
        sizes.append(parts[6])
        filters.append(parts[7])

    return {
        'filenames': filenames,
        'sites': np.array(sites),
        'ccds': np.array(ccds),
        'serials': np.array(serials),
        'nights': np.array(nights),
        'binnings': np.array(binnings),
        'sizes': np.array(sizes),
        'filters': np.array(filters),
        'times': np.array([datetime.datetime.strptime(night, '%Y%m%d') for night in nights]),
    }


def load_std_metrics(filenames):
    std_means = []
    std_medians = []
    std_rms_values = []

    for filename in tqdm(filenames, desc='Reading nightly flat stats'):
        header = fits.getheader(filename.replace('_min', '_std'), -1)
        std_means.append(header['STDMEAN'])
        std_medians.append(header['STDMED'])
        std_rms_values.append(header['STDRMS'])

    return {
        'stdmeans': np.array(std_means),
        'stdmedians': np.array(std_medians),
        'stdstds': np.array(std_rms_values),
    }


def print_unique_configs(metadata):
    sites = metadata['sites']
    ccds = metadata['ccds']
    serials = metadata['serials']
    binnings = metadata['binnings']
    sizes = metadata['sizes']
    filters = metadata['filters']
    nights = metadata['nights']
    filenames = metadata['filenames']

    for site in np.unique(sites):
        site_mask = sites == site

        for ccd in np.unique(ccds[site_mask]):
            ccd_mask = site_mask & (ccds == ccd)

            for serial in np.unique(serials[ccd_mask]):
                serial_mask = ccd_mask & (serials == serial)

                for binning in np.unique(binnings[serial_mask]):
                    binning_mask = serial_mask & (binnings == binning)

                    for size in np.unique(sizes[binning_mask]):
                        size_mask = binning_mask & (sizes == size)

                        for filter_name in np.unique(filters[size_mask]):
                            selection_mask = size_mask & (filters == filter_name)
                            print(
                                site,
                                ccd,
                                serial,
                                binning,
                                size,
                                filter_name,
                                ':',
                                len(filenames[selection_mask]),
                                'per-night flats,',
                                nights[selection_mask][0],
                                '-',
                                nights[selection_mask][-1],
                            )


def get_config_mask(metadata, cfg):
    mask = metadata['sites'] == cfg.get('site')
    mask &= metadata['ccds'] == cfg.get('ccd')
    mask &= metadata['serials'] == cfg.get('serial')

    binning = cfg.get('binning', '1x1')
    mask &= metadata['binnings'] == binning

    size = cfg.get('size')
    if size:
        mask &= metadata['sizes'] == size

    if 'first' in cfg:
        mask &= metadata['nights'] >= cfg.get('first')
    if 'last' in cfg:
        mask &= metadata['nights'] <= cfg.get('last')

    return mask


def build_masterflat_name(basedir, site, ccd, serial, night, binning, size, filter_name):
    return os.path.join(
        basedir,
        site,
        'masterflats',
        'masterflat_%s_%s_%s_%s_%s_%s_%s.fits' % (
            site,
            ccd,
            serial,
            night,
            binning,
            size,
            filter_name,
        ),
    )


def compute_std_threshold(std_values):
    threshold = np.inf
    keep_mask = np.ones_like(std_values, dtype=bool)

    for _ in range(10):
        current_values = std_values[keep_mask]
        if not len(current_values):
            break

        threshold = 3.0 * rstd(current_values) + rmean(current_values)
        new_mask = std_values < threshold
        if not np.any(new_mask):
            break

        keep_mask = new_mask

    return threshold


def get_stable_nightly_selection(metadata, metrics, selection_mask):
    std_threshold = compute_std_threshold(metrics['stdstds'][selection_mask])
    stable_mask = metrics['stdstds'][selection_mask] <= std_threshold

    filenames = metadata['filenames'][selection_mask][stable_mask]
    nights = metadata['nights'][selection_mask][stable_mask]
    order = np.argsort(nights)

    return filenames[order], nights[order], std_threshold


def iter_monthly_shards(nights, min_flats):
    if not len(nights):
        return

    segment_start_night = nights[0]
    segment_end_night = segment_start_night
    last_night = nights[-1]

    while True:
        if segment_start_night > last_night or segment_end_night > last_night:
            break

        segment_end_night = get_next_month(segment_end_night)
        segment_mask = (nights >= segment_start_night) & (nights < segment_end_night)

        if np.sum(segment_mask) >= min_flats:
            yield segment_start_night, segment_mask
            segment_start_night = segment_end_night


def prepare_output_header(header, site, night):
    header = header.copy()
    header['IMAGETYP'] = 'masterflat'
    header['DATE-OBS'] = get_iso_time(get_night_time(night, lon=header.get('LONGITUD'), site=site))
    header['DATE'] = get_iso_time(datetime.datetime.now(datetime.timezone.utc))
    return header


def coadd_images(filenames, progress_label):
    coadd = None
    header = None
    n_used = 0

    for filename in tqdm(filenames, leave=False, desc=progress_label):
        if header is None:
            header = fits.getheader(filename, -1)

        image = fits.getdata(filename, -1).astype(np.double)
        image /= np.median(image)

        coadd = image if coadd is None else coadd + image
        n_used += 1

    return coadd, header, n_used


def process_every_mode(site, ccd, serial, binning, size, filter_name, basedir, replace, filenames, nights):
    for index, filename in enumerate(tqdm(filenames, leave=False, desc=filter_name)):
        nightly_output_name = build_masterflat_name(
            basedir,
            site,
            ccd,
            serial,
            nights[index],
            binning,
            size,
            filter_name,
        )

        if os.path.exists(nightly_output_name) and not replace:
            continue

        header = fits.getheader(filename, -1)
        image = fits.getdata(filename, -1).astype(np.double)
        image /= np.median(image)

        output_header = prepare_output_header(header, site, nights[index])
        fits_write(nightly_output_name, image, output_header, compress=True)


def process_coadd_mode(site, ccd, serial, binning, size, filter_name, basedir, replace, filenames, nights, min_flats):
    wrote_any = False

    for segment_start_night, segment_mask in iter_monthly_shards(nights, min_flats):
        flatname = build_masterflat_name(
            basedir,
            site,
            ccd,
            serial,
            segment_start_night,
            binning,
            size,
            filter_name,
        )
        os.makedirs(os.path.dirname(flatname), exist_ok=True)

        print(flatname)
        if os.path.exists(flatname) and not replace:
            wrote_any = True
            continue

        segment_filenames = filenames[segment_mask]
        segment_nights = nights[segment_mask]

        print(
            site,
            ccd,
            serial,
            binning,
            size,
            filter_name,
            ':',
            len(segment_filenames),
            'nightly flats,',
            segment_nights[0],
            '-',
            segment_nights[-1],
        )

        coadd, header, n_used = coadd_images(segment_filenames, filter_name)
        if n_used == 0:
            continue

        flat = coadd / np.median(coadd)
        output_header = prepare_output_header(header, site, segment_start_night)
        fits_write(flatname, flat, output_header, compress=True)
        print(flatname)
        wrote_any = True

    return wrote_any


def process_filter_group(cfg, metadata, metrics, selection_mask, filter_name, basedir, replace, min_flats):
    site = cfg.get('site')
    ccd = cfg.get('ccd')
    serial = cfg.get('serial')
    binning = cfg.get('binning', '1x1')
    size = metadata['sizes'][selection_mask][0]
    mode = cfg.get('mode', 'coadd')

    print(
        site,
        ccd,
        serial,
        binning,
        size,
        filter_name,
        ':',
        len(metadata['filenames'][selection_mask]),
        'per-night flats,',
        metadata['nights'][selection_mask][0],
        '-',
        metadata['nights'][selection_mask][-1],
    )

    stable_filenames, stable_nights, std_threshold = get_stable_nightly_selection(
        metadata,
        metrics,
        selection_mask,
    )
    if not len(stable_filenames):
        print('No stable nightly flats after clipping for', site, ccd, serial, filter_name)
        return

    if mode == 'every':
        process_every_mode(
            site,
            ccd,
            serial,
            binning,
            size,
            filter_name,
            basedir,
            replace,
            stable_filenames,
            stable_nights,
        )
        return

    if not process_coadd_mode(
        site,
        ccd,
        serial,
        binning,
        size,
        filter_name,
        basedir,
        replace,
        stable_filenames,
        stable_nights,
        min_flats,
    ):
        print(
            'Not enough stable nightly flats for',
            site,
            ccd,
            serial,
            filter_name,
            'with min_flats =',
            min_flats,
        )

def process_config(cfg, metadata, metrics, basedir, replace, min_flats):
    config_mask = get_config_mask(metadata, cfg)
    if not np.any(config_mask):
        return

    sizes = np.unique(metadata['sizes'][config_mask])
    if len(sizes) != 1:
        raise RuntimeError('%s %s' % (cfg, sizes))

    for filter_name in np.unique(metadata['filters'][config_mask]):
        filter_mask = config_mask & (metadata['filters'] == filter_name)
        process_filter_group(
            cfg,
            metadata,
            metrics,
            filter_mask,
            filter_name,
            basedir,
            replace,
            min_flats,
        )


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage='usage: %prog [options]')

    parser.add_option(
        '-g',
        '--glob',
        help='Glob pattern for nightly flat _min files',
        action='store',
        dest='pattern',
        type='str',
        default='calibrations/*/flats/*_min.fits',
    )
    parser.add_option(
        '-B',
        '--basedir',
        help='Base directory for output files',
        action='store',
        dest='basedir',
        type='str',
        default='calibrations',
    )
    parser.add_option(
        '-r',
        '--replace',
        help='Replace existing files',
        action='store_true',
        dest='replace',
        default=False,
    )
    parser.add_option(
        '--list-configs',
        help='Print unique nightly-flat configurations and exit',
        action='store_true',
        dest='list_configs',
        default=False,
    )
    parser.add_option(
        '--min-flats',
        help='Minimal number of nightly flats per coadded masterflat shard',
        action='store',
        dest='min_flats',
        type='int',
        default=10,
    )

    options, args = parser.parse_args()

    metadata = load_nightly_flat_metadata(options.pattern)
    if metadata is None:
        print('No nightly flat files found for pattern %s' % options.pattern, file=sys.stderr)
        sys.exit(0)

    print(len(metadata['filenames']), 'nightly flat files found', file=sys.stderr)

    if options.list_configs:
        print_unique_configs(metadata)
        sys.exit(0)

    metrics = load_std_metrics(metadata['filenames'])

    for cfg in flat_configs:
        process_config(
            cfg,
            metadata,
            metrics,
            options.basedir,
            options.replace,
            options.min_flats,
        )
