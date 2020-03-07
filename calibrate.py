from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from scipy.stats import sigmaclip

# Configurations
calibration_configs = [
    # Old and unsupported?..
    # Auger NF3
    {'serial':2328, 'binning':'1x1', 'date-before':'2017-07-01', 'means_min':350, 'means_max':500, 'airtemp_a': -0.46159676, 'airtemp_b':423.75726376},
    {'serial':2328, 'binning':'1x1', 'date-after':'2017-07-01', 'means_min':350, 'means_max':500, 'airtemp_a':-0.62109129, 'airtemp_b':398.78857177},
    # WF5 - no science frames?..
    # {'serial':3072, 'binning':'1x1'},

    # La Palma custom G2
    {'serial':2596, 'binning':'1x1', 'airtemp_a':-0.2576, 'airtemp_b':478.2},

    # 6029 before recalibration, NO OVERSCAN DATA! NO LINEARIZATION DATA!
    {'serial':6029, 'binning':'1x1', 'date-before':'2017-10-01', 'airtemp_a':0.41, 'airtemp_b':162.3, 'param1':[0.07290866, 0.81854223, 0.1754477 ,-0.55883105], 'param2':3.173551818062375},
    # 6029 after recalibration, NO OVERSCAN DATA!
    {'serial':6029, 'binning':'1x1', 'date-after':'2017-10-01', 'date-before':'2018-09-01', 'airtemp_a':0.78, 'airtemp_b':501, 'param1':[0.07290866, 0.81854223, 0.1754477 ,-0.55883105], 'param2':3.173551818062375},
    # 6029 after board upgrade
    {'serial':6029, 'binning':'1x1', 'date-after':'2019-03-01', 'airtemp_a':0.61, 'airtemp_b':502.8, 'param1':[0.05545345, 0.86168363, 0.14405999,-0.45671358], 'param2':3.1756676670297868},

    # 6069 before board upgrade, NO OVERSCAN DATA!
    {'serial':6069, 'binning':'1x1', 'date-before':'2018-08-01', 'airtemp_a':0.487, 'airtemp_b':508.0, 'param1':[0.07484702, 0.79481961, 0.14635468,-0.43780121], 'param2':2.990236610791621},
    # 6069 after board upgrade
    {'serial':6069, 'binning':'1x1', 'date-after':'2018-08-01', 'airtemp_a':-2.0, 'airtemp_b':585.7, 'param1':[0.07433374, 0.79046048, 0.0810416 ,-0.25420098], 'param2':3.137078318644374},

    # 6132
    {'serial':6132, 'binning':'1x1', 'airtemp_a':0.516, 'airtemp_b':551.5, 'param1':[0.05483907, 0.84237167, 0.0629831 ,-0.18487527], 'param2':2.9551997417641154},

    # 6149
    {'serial':6149, 'binning':'1x1', 'airtemp_a':0.85, 'airtemp_b':503.6, 'param1':[0.03364724, 0.90427548, 0.07500224,-0.21427316], 'param2':2.8578692493195548},

    # 6166 / SBT
    {'serial':6166, 'binning':'1x1', 'param1':[0.07392971, 0.77232407,-0.01752741, 0.01464123], 'param2':2.111340867505684},

    # 6166 / SBT
    {'serial':6167, 'binning':'1x1', 'param1':[0.10854498, 0.67574182, 0.02173521,-0.08850012], 'param2':2.6305813378315057},

    # 6204 before readout upgrade
    {'serial':6204, 'binning':'1x1', 'date-before':'2019-12-01', 'airtemp_a':-1.55, 'airtemp_b':599.4, 'param1':[0.05399309, 0.84056592, 0.02539838,-0.06620873], 'param2':2.602603959616151},
    {'serial':6204, 'binning':'2x2', 'date-before':'2019-12-01', 'airtemp_a':-1.51, 'airtemp_b':698.7},
    # 6204 after readout upgrade, seems nonlin did not change
    {'serial':6204, 'binning':'1x1', 'date-after':'2019-12-01', 'date-before':'2020-02-18', 'airtemp_a':-1.42, 'airtemp_b':574.1, 'param1':[0.05399309, 0.84056592, 0.02539838,-0.06620873], 'param2':2.602603959616151},
    # 6204, preflash
    {'serial':6204, 'binning':'1x1', 'date-after':'2020-02-17', 'airtemp_a':-1.42, 'airtemp_b':574.1, 'param1':[0.05399309, 0.84056592, 0.02539838,-0.06620873], 'param2':2.602603959616151},

    # 6205
    {'serial':6205, 'binning':'1x1', 'airtemp_a':-1.50, 'airtemp_b':584.0, 'param1':[0.04765044, 0.85458389,-0.03480524, 0.09953474], 'param2':2.8619181700630127},
    {'serial':6205, 'binning':'2x2', 'airtemp_a':-1.45, 'airtemp_b':698, 'date-before':'2019-10-22'},
    {'serial':6205, 'binning':'2x2', 'airtemp_a':-1.33889045, 'airtemp_b':703.72816149, 'date-after':'2019-10-21'},
]

# Selecting proper calibration config
def find_calibration_config(header=None, serial=None, binning=None, width=None, height=None, date=None):
    if header is None:
        header = {'product_id':serial, 'BINNING':binning, 'NAXIS1':width, 'NAXIS2':height, 'DATE-OBS':date}

    for cfg in calibration_configs:
        if cfg.has_key('serial') and cfg.get('serial') != header['product_id']:
            continue

        if cfg.has_key('binning') and cfg.get('binning') != header['BINNING']:
            continue

        if cfg.has_key('width') and cfg.get('width') != header['NAXIS1']:
            continue

        if cfg.has_key('height') and cfg.get('height') != header['NAXIS2']:
            continue

        if cfg.has_key('date-before') and cfg.get('date-before') < header['DATE-OBS']:
            continue

        if cfg.has_key('date-after') and cfg.get('date-after') > header['DATE-OBS']:
            continue

        return cfg

    return None

# Robust mean
def rmean(data, max=None):
    if max is not None:
        data = np.asarray(data)
        data = data[data<max]
    return np.mean(sigmaclip(data, 3.0, 3.0)[0])

def rstd(data):
    return 1.4826*np.median(np.abs(data - np.median(data)))

# Parsing of DATASEC-like keywords
def parse_det(string):
    '''Parse DATASEC-like keyword'''
    x0,x1,y0,y1 = [int(_)-1 for _ in sum([_.split(':') for _ in string[1:-1].split(',')], [])]

    return x0,x1,y0,y1

def get_cropped_shape(shape=None, header=None):
    '''Get the shape of an image after overscan cropping based on its header or, well, shape'''
    if shape is None and header is not None:
        shape = (header['NAXIS2'], header['NAXIS1'])

    if header is None or not header.get('DATASEC'):
        if shape == (4124, 4148) or shape == (4127,4144):
            result = (4096, 4096)
        elif shape == (2062, 2074) or shape == (2063, 2072):
            result = (2048, 2048)
        elif shape == (1026, 1062):
            result = (1024, 1056)
        else:
            result = shape
    elif header.get('DATASEC'):
        x1,x2,y1,y2 = parse_det(header.get('DATASEC'))
        result = (y2-y1+1, x2-x1+1)
    else:
        result = shape

    return result

# Cropping of overscans if any
def crop_overscans(image, header=None, subtract=True, cfg=None):
    ''''Crop overscans from input image based on its header or dimensions.
    Also, subtract the 'bias' value estimated from either an overscan or pre-determined temperature trend.
    '''
    if header is not None:
        header = header.copy()

    # Estimate bias level from overscan
    bias = None

    if image.shape == (4124, 4148): # Initial patched G4 firmware
        bias = rmean(list(image[2:8, 300:-300].flatten()) + list(image[-14:, 300:-300].flatten()))

    elif image.shape == (2062, 2074) and header.get('BINNING') == '2x2': # The same, 2x2 binning
        bias = rmean(list(image[1:4, 150:-150].flatten()) + list(image[-7:, 150:-150].flatten()))

    elif image.shape == (4127,4144): # Official firmwares after enabling overscans in Windows utility
        bias = rmean(list(image[3:7, 300:-300].flatten()) + list(image[-17:-4, 300:-300].flatten()))

    elif image.shape == (2063, 2072) and header.get('BINNING') == '2x2': # The same, 2x2 binning
        bias = rmean(list(image[2:3, 150:-150].flatten()) + list(image[-8:-2, 150:-150].flatten()))

    elif image.shape == (1026, 1062): # Overscan-enabled custom G2 on La Palma
        bias = rmean(list(image[:, -5:]))

    elif header and header.get('product_id') >= 6000 and header.get('SHIFT') is not None and image.shape == (4096, 4096):
        # Ultra-special handling of a MICCD frames what are vertically flipped in respect to GXCCD ones
        image = image[::-1, :] # Flip the image

        if header.get('CRPIX1') is not None:
            header['CRPIX2'] = header['NAXIS2'] + 1 - header['CRPIX2']
            header['CD1_2'] *= -1
            header['CD2_2'] *= -1

            if header.get('A_ORDER') and header.get('B_ORDER'):
                for p in xrange(max(header['A_ORDER'], header['B_ORDER'])+1):
                    for q in xrange(max(header['A_ORDER'], header['B_ORDER'])+1):
                        if (q % 2) and header.get('A_%d_%d' % (p,q)):
                            header['A_%d_%d' % (p,q)] *= -1
                        if (q % 2) == 0 and header.get('B_%d_%d' % (p,q)):
                            header['B_%d_%d' % (p,q)] *= -1

    if bias is not None:
        if header is not None:
            # Store overscan-estimated bias level to FITS header
            header['BIASAVG'] = bias
    else:
        # No overscans, let's try to fit it if we have a model for this configuration
        if cfg is None:
            cfg = find_calibration_config(header)

        if cfg and cfg.has_key('airtemp_a') and cfg.has_key('airtemp_b'):
            bias = header['CCD_AIR']*cfg['airtemp_a'] + cfg['airtemp_b']

    if bias is not None and subtract:
            image = image.copy() - bias

    if header is None or not header.get('DATASEC'):
        # Special handling of legacy G4 data - manually adjusted overscan-free regions
        if image.shape == (4124, 4148): # Initial patched G4 firmware
            image = image[11:-17, 33:-19]

            if header is not None and header.get('CRPIX1') is not None:
                header['CRPIX1'] -= 33
                header['CRPIX2'] -= 11
            if header is not None:
                header['DATASEC0'] = '[34:4129,12:4107]'
        elif image.shape == (2062, 2074) and header.get('BINNING') == '2x2': # The same, 2x2 binning
            image = image[5:-9, 17:-9]

            if header is not None and header.get('CRPIX1') is not None:
                header['CRPIX1'] -= 15
                header['CRPIX2'] -= 5
            if header is not None:
                header['DATASEC0'] = '[18:2065,6:2053]'

        elif image.shape == (4127,4144): # Official firmwares after enabling overscans in Windows utility
            image = image[11:-20, 30:-18]

            if header is not None and header.get('CRPIX1') is not None:
                header['CRPIX1'] -= 30
                header['CRPIX2'] -= 11
            if header is not None:
                header['DATASEC0'] = '[31:4126,12:4107]'
        elif image.shape == (2063, 2072) and header.get('BINNING') == '2x2': # The same, 2x2 binning
            image = image[5:-10, 15:-9]

            if header is not None and header.get('CRPIX1') is not None:
                header['CRPIX1'] -= 15
                header['CRPIX2'] -= 5
            if header is not None:
                header['DATASEC0'] = '[16:2063,6:2053]'

        elif image.shape == (1026, 1062): # Overscan-enabled custom G2 on La Palma
            image = image[2:, 0:1056]

            if header is not None and header.get('CRPIX1') is not None:
                header['CRPIX1'] -= 0
                header['CRPIX2'] -= 2
            if header is not None:
                header['DATASEC0'] = '[1:1056,3:1026]'

    else:
        x1,x2,y1,y2 = parse_det(header.get('DATASEC'))

        if header is not None and header.get('CRPIX1') is not None:
            header['CRPIX1'] -= x1
            header['CRPIX2'] -= y1

        image = image[y1:y2+1, x1:x2+1]
        header['DATASEC0'] = header.pop('DATASEC')

    image = np.ascontiguousarray(image)

    if header is not None:
        # Update NAXIS keywords to reflect cropped dimensions of the image
        header['NAXIS1'] = image.shape[1]
        header['NAXIS2'] = image.shape[0]

        return image,header
    else:
        return image

def calibrate(image, header, dark=None, crop=True, subtract=True, linearize=True):
    '''Higher-level image calibration based on its header.
    Includes overscan cropping and subtraction, bias subtraction and linearization.'''
    cfg = find_calibration_config(header)

    if crop:
        image,header = crop_overscans(image, header, subtract=subtract, cfg=cfg)

    if dark is not None:
        if dark.shape != image.shape:
            print("Wrong dark shape:", dark.shape, "vs", image.shape)
        else:
            image = 1.0*image - dark

    if linearize:
        if cfg and cfg.has_key('param1') and cfg.has_key('param2'):
            param1, param2 = cfg['param1'], cfg['param2']

        else:
            # print("No linearization for this chip")
            param1, param2 = [0, 1, 0, 0], 0

        # Keep linearization parmeters in the header
        header = header.copy()
        for _ in [0,1,2,3]:
            header['PARAM1_%d' % _] = param1[_]
        header['PARAM2'] = param2

        image = image.copy()
        B = np.log10(image[image > 1])
        v1 = param1[0]*B + param1[1]
        v2 = param1[2]*B + param1[3]
        v1[B < param2] += v2[B < param2]

        image[image > 1] /= v1

    return image, header
