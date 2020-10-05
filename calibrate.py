from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from scipy.stats import sigmaclip
from scipy.interpolate import interp1d

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
    {'serial':6029, 'binning':'1x1', 'date-before':'2017-10-01', 'airtemp_a':0.41, 'airtemp_b':162.3,
     # Linearization from next config!!!
     'points':[[0.67844342, 0.53257576], [1.07846902, 0.57424242], [1.43049155, 0.62348485], [1.71850998, 0.67146465], [2.00652842, 0.74280303], [2.2593446 , 0.81792929], [2.47695853, 0.88484848], [2.70097286, 0.94671717], [2.88658474, 0.99532828], [3.0593958 , 1.03636364], [3.28341014, 1.06540404], [3.46902202, 1.08434343], [3.73783922, 1.10707071], [3.95545315, 1.11906566], [4.1890681 , 1.13421717], [4.390681 , 1.13926768], [4.59869432, 1.14431818], [4.59869432, 1.14431818], [4.73950333, 1.13674242], [4.79390681, 1.12727273]]},
    # 6029 after recalibration, NO OVERSCAN DATA!
    {'serial':6029, 'binning':'1x1', 'date-after':'2017-10-01', 'date-before':'2018-09-01', 'airtemp_a':0.78, 'airtemp_b':501,
     'points':[[0.67844342, 0.53257576], [1.07846902, 0.57424242], [1.43049155, 0.62348485], [1.71850998, 0.67146465], [2.00652842, 0.74280303], [2.2593446 , 0.81792929], [2.47695853, 0.88484848], [2.70097286, 0.94671717], [2.88658474, 0.99532828], [3.0593958 , 1.03636364], [3.28341014, 1.06540404], [3.46902202, 1.08434343], [3.73783922, 1.10707071], [3.95545315, 1.11906566], [4.1890681 , 1.13421717], [4.390681 , 1.13926768], [4.59869432, 1.14431818], [4.59869432, 1.14431818], [4.73950333, 1.13674242], [4.79390681, 1.12727273]]},
    # 6029 after board upgrade
    {'serial':6029, 'binning':'1x1', 'date-after':'2019-03-01', 'airtemp_a':0.61, 'airtemp_b':502.8,
     'points':[[0.79365079, 0.66073232], [1.28968254, 0.70176768], [1.57450077, 0.72828283], [1.9297235 , 0.78194444], [2.20494112, 0.84318182], [2.42895545, 0.89368687], [2.67857143, 0.9478938 ], [2.92050691, 0.99574069], [3.14900154, 1.0281077 ], [3.406298 , 1.05109297], [3.69239631, 1.07173281], [4.06490015, 1.08815086], [4.3452381 , 1.09893986], [4.60829493, 1.10316165], [4.71198157, 1.10034712], [4.80414747, 1.08908903]]},

    # 6069 before board upgrade, NO OVERSCAN DATA!
    {'serial':6069, 'binning':'1x1', 'date-before':'2018-08-01', 'airtemp_a':0.487, 'airtemp_b':508.0,
     'points':[[0.97606247, 0.67083333], [1.60970302, 0.72323232], [1.76651306, 0.74785354], [1.95212494, 0.78888889], [2.2593446 , 0.85328283], [2.4577573 , 0.90378788], [2.70417307, 0.95871212], [2.88338454, 0.99343434], [3.05299539, 1.01742424], [3.29621096, 1.04078283], [3.69303635, 1.07234848], [3.98105479, 1.08876263], [4.42268305, 1.12222222], [4.65949821, 1.13484848], [4.73950333, 1.13484848], [4.79710701, 1.12790404]]},
    # 6069 after board upgrade
    {'serial':6069, 'binning':'1x1', 'date-after':'2018-08-01', 'date-before':'2020-03-29', 'airtemp_a':-2.0, 'airtemp_b':585.7,
     'points':[[0.74244752, 0.70555556], [1.07526882, 0.72449495], [1.42729135, 0.7510101 ], [1.71210958, 0.79141414], [2.02892985, 0.84318182], [2.30094726, 0.88800505], [2.56336406, 0.9385101 ], [2.81938044, 0.98712121], [3.12980031, 1.02626263], [3.40501792, 1.04835859], [3.71543779, 1.07234848], [4.00985663, 1.09255051], [4.28507424, 1.11338384], [4.47708653, 1.12916667], [4.66589862, 1.1405303 ], [4.74910394, 1.14116162], [4.80350742, 1.13611111]]},
    # 6069, preflash
    {'serial':6069, 'binning':'1x1', 'date-after':'2020-03-28', 'airtemp_a':-2.0, 'airtemp_b':585.7,
     'points':[[0.74244752, 0.70555556], [1.07526882, 0.72449495], [1.42729135, 0.7510101 ], [1.71210958, 0.79141414], [2.02892985, 0.84318182], [2.30094726, 0.88800505], [2.56336406, 0.9385101 ], [2.81938044, 0.98712121], [3.12980031, 1.02626263], [3.40501792, 1.04835859], [3.71543779, 1.07234848], [4.00985663, 1.09255051], [4.28507424, 1.11338384], [4.47708653, 1.12916667], [4.66589862, 1.1405303 ], [4.74910394, 1.14116162], [4.80350742, 1.13611111]]},

    # 6132 - NO OVERSCAN DATA!
    {'serial':6132, 'binning':'1x1', 'airtemp_a':0.516, 'airtemp_b':551.5,
     'points':[[0.61443932, 0.7895202 ], [1.04646697, 0.8040404 ], [1.38248848, 0.82424242], [1.70250896, 0.85328283], [1.89452125, 0.87727273], [2.09933436, 0.90189394], [2.390553 , 0.93977273], [2.61776754, 0.96691919], [2.88018433, 0.99406566], [3.18100358, 1.01489899], [3.4562212 , 1.02752525], [3.73463902, 1.04204545], [3.9874552 , 1.05782828], [4.27867384, 1.07676768], [4.47068612, 1.09381313], [4.60189452, 1.1020202 ], [4.68509985, 1.10328283], [4.75230415, 1.09823232]]},

    # 6149
    {'serial':6149, 'binning':'1x1', 'airtemp_a':0.85, 'airtemp_b':503.6,
     'points':[[0.62403994, 0.77941919], [1.24807988, 0.82487374], [1.56169995, 0.85643939], [1.87532002, 0.89116162], [2.13453661, 0.92335859], [2.50576037, 0.96502525], [2.85458269, 0.99659091], [3.20980543, 1.01426768], [3.54262673, 1.025 ], [3.89464926, 1.03510101], [4.23707117, 1.04772727], [4.55389145, 1.05719697], [4.70110087, 1.0540404 ], [4.7843062 , 1.04646465]]},

    # 6166 / SBT
    {'serial':6166, 'binning':'1x1',
     'points':[[1.00166411, 0.83939394], [1.53289811, 0.86338384], [1.78251408, 0.88042929], [2.28814644, 0.93409091], [2.76817716, 0.98712121], [3.07219662, 1.00795455], [3.47542243, 1.02941919], [3.82744496, 1.04393939], [4.19546851, 1.05782828], [4.51228879, 1.06792929], [4.69790067, 1.06792929], [4.81310804, 1.05972222]]},

    # 6167 / SBT
    {'serial':6167, 'binning':'1x1',
     'point':[[0.96646185, 0.72449495], [1.53289811, 0.77121212], [1.67370712, 0.78825758], [2.33934972, 0.89810606], [2.74897593, 0.96881313], [3.01779314, 1.00416667], [3.54262673, 1.06098485], [4.04185868, 1.11022727], [4.41628264, 1.15063131], [4.5890937 , 1.1645202 ], [4.74270353, 1.16515152], [4.79710701, 1.15694444]]},

    # 6204 before readout upgrade
    {'serial':6204, 'binning':'1x1', 'date-before':'2019-12-01', 'airtemp_a':-1.55, 'airtemp_b':599.4,
     'points':[[0.94406042, 0.82424242], [1.45929339, 0.88169192], [1.87532002, 0.92335859], [2.27534562, 0.95555556], [2.6593702 , 0.98333333], [3.04339478, 1.00606061], [3.3922171 , 1.02626263], [3.80184332, 1.05025253], [4.13786482, 1.07739899], [4.44828469, 1.10265152], [4.656298 , 1.11590909], [4.73950333, 1.11717172], [4.80350742, 1.11338384]]},
    {'serial':6204, 'binning':'2x2', 'date-before':'2019-12-01', 'airtemp_a':-1.51, 'airtemp_b':698.7},
    # 6204 after readout upgrade, seems nonlin did not change
    {'serial':6204, 'binning':'1x1', 'date-after':'2019-12-01', 'date-before':'2020-02-18', 'airtemp_a':-1.42, 'airtemp_b':574.1,
     'points':[[1.42729135, 0.88106061], [2.01932924, 0.93472222], [2.40975422, 0.96439394], [2.78417819, 0.99027778], [3.05299539, 1.00542929], [3.44022017, 1.025 ], [3.75064004, 1.03825758], [3.92985151, 1.04962121], [4.14746544, 1.06477273], [4.35227855, 1.08244949], [4.50588838, 1.09444444], [4.66909882, 1.10517677], [4.75230415, 1.10454545], [4.79710701, 1.09823232]]},
    # 6204, preflash
    {'serial':6204, 'binning':'1x1', 'date-after':'2020-02-17', 'airtemp_a':-1.42, 'airtemp_b':574.1,
     'points':[[1.42729135, 0.88106061], [2.01932924, 0.93472222], [2.40975422, 0.96439394], [2.78417819, 0.99027778], [3.05299539, 1.00542929], [3.44022017, 1.025 ], [3.75064004, 1.03825758], [3.92985151, 1.04962121], [4.14746544, 1.06477273], [4.35227855, 1.08244949], [4.50588838, 1.09444444], [4.66909882, 1.10517677], [4.75230415, 1.10454545], [4.79710701, 1.09823232]]},

    # 6205
    {'serial':6205, 'binning':'1x1', 'airtemp_a':-1.50, 'airtemp_b':584.0,
     'points':[[0.65604199, 0.95618687], [1.35688684, 0.96439394], [1.80491551, 0.97638889], [2.16013825, 0.98017677], [2.54736303, 0.98396465], [2.73937532, 0.99217172], [3.09459805, 1.00921717], [3.60023041, 1.02752525], [4.10906298, 1.05530303], [4.43548387, 1.08244949], [4.67549923, 1.09760101], [4.74910394, 1.09760101], [4.81310804, 1.09255051]]},
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
        # bias = rmean(list(image[2:8, 300:-300].flatten()) + list(image[-14:, 300:-300].flatten()))
        bias = rmean(image[-14:-4, 800:-800].flatten())

    elif image.shape == (2062, 2074) and header.get('BINNING') == '2x2': # The same, 2x2 binning
        bias = rmean(image[-7:-2, 400:-400].flatten())

    elif image.shape == (4127,4144): # Official firmwares after enabling overscans in Windows utility
        # bias = rmean(list(image[3:7, 300:-300].flatten()) + list(image[-17:-4, 300:-300].flatten()))
        bias = rmean(image[-14:-4, 800:-800].flatten())

    elif image.shape == (2063, 2072) and header.get('BINNING') == '2x2': # The same, 2x2 binning
        bias = rmean(image[-7:-2, 400:-400].flatten())

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

    header = header.copy()

    # Sanitize GAIN value from La Palma custom G2 (MICCD driver)
    if 'GAIN' in header and header['GAIN'] > 1000:
        header['GAIN'] = 1e-3*header['GAIN']

    if linearize:
        if cfg and cfg.has_key('points'):
            points = np.array(cfg['points'])
            image = image.copy()
            image /= interp1d(10**points[:,0], points[:,1], fill_value='extrapolate')(image)

        else:
            if cfg and cfg.has_key('param1') and cfg.has_key('param2'):
                param1, param2 = cfg['param1'], cfg['param2']

            else:
                # print("No linearization for this chip")
                param1, param2 = [0, 1, 0, 0], 0

            # Keep linearization parmeters in the header
            for _ in [0,1,2,3]:
                header['PARAM1_%d' % _] = param1[_]
            header['PARAM2'] = param2

            image = image.copy()
            B = np.log10(image[image > 1])
            v1 = param1[0]*B + param1[1]
            v2 = param1[2]*B + param1[3]
            v1[B < param2] += v2[B < param2]

            image[image > 1] /= v1

        header['LINEARIZ'] = 1

    return image, header
