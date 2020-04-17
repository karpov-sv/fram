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
    {'serial':6029, 'binning':'1x1', 'date-before':'2017-10-01', 'airtemp_a':0.41, 'airtemp_b':162.3, 'param1':[0.07290866, 0.81854223, 0.1754477 ,-0.55883105], 'param2':3.173551818062375,
     'points':[[0.9952637 , 0.55467172], [1.37288786, 0.60896465], [1.65770609, 0.65441919], [2.08333333, 0.75479798], [2.51536098, 0.88232323], [2.84818228, 0.98333333], [3.24180748, 1.0489899 ], [3.70583717, 1.09633838], [4.17306708, 1.12411616], [4.59229391, 1.13421717], [4.71070148, 1.13042929], [4.80990783, 1.11527778]]},
    # 6029 after recalibration, NO OVERSCAN DATA!
    {'serial':6029, 'binning':'1x1', 'date-after':'2017-10-01', 'date-before':'2018-09-01', 'airtemp_a':0.78, 'airtemp_b':501, 'param1':[0.07290866, 0.81854223, 0.1754477 ,-0.55883105], 'param2':3.173551818062375,
     'points':[[0.9952637 , 0.55467172], [1.37288786, 0.60896465], [1.65770609, 0.65441919], [2.08333333, 0.75479798], [2.51536098, 0.88232323], [2.84818228, 0.98333333], [3.24180748, 1.0489899 ], [3.70583717, 1.09633838], [4.17306708, 1.12411616], [4.59229391, 1.13421717], [4.71070148, 1.13042929], [4.80990783, 1.11527778]]},
    # 6029 after board upgrade
    {'serial':6029, 'binning':'1x1', 'date-after':'2019-03-01', 'airtemp_a':0.61, 'airtemp_b':502.8, 'param1':[0.05545345, 0.86168363, 0.14405999,-0.45671358], 'param2':3.1756676670297868,
     'points':[[0.58279707, 0.60283604], [1.33390045, 0.68829929], [1.53675319, 0.71605282], [1.74874344, 0.74600901], [2.07221132, 0.80856458], [2.35730166, 0.87464441], [2.52725936, 0.91693551], [2.82879721, 0.97905055], [3.08099251, 1.01957951], [3.38070286, 1.0495357 ], [3.70051574, 1.07200284], [4.09891121, 1.0891836 ], [4.5210642 , 1.10460223], [4.62157682, 1.10548329], [4.79884453, 1.09138626]]},

    # 6069 before board upgrade, NO OVERSCAN DATA!
    {'serial':6069, 'binning':'1x1', 'date-before':'2018-08-01', 'airtemp_a':0.487, 'airtemp_b':508.0, 'param1':[0.07484702, 0.79481961, 0.14635468,-0.43780121], 'param2':2.990236610791621,
     'points':[[0.94726062, 0.67840909], [1.7313108 , 0.73712121], [1.96492576, 0.78320707], [2.29774706, 0.8614899 ], [2.48655914, 0.90820707], [2.71377368, 0.96123737], [2.82258065, 0.98522727], [3.08179724, 1.02058081], [3.51062468, 1.0540404 ], [3.88184844, 1.07992424], [4.51548899, 1.12979798], [4.67549923, 1.13358586], [4.80990783, 1.12537879]]},
    # 6069 after board upgrade
    {'serial':6069, 'binning':'1x1', 'date-after':'2018-08-01', 'date-before':'2020-03-29', 'airtemp_a':-2.0, 'airtemp_b':585.7, 'param1':[0.07433374, 0.79046048, 0.0810416 ,-0.25420098], 'param2':3.137078318644374,
     'points':[[1.00486431, 0.71944444], [1.37928827, 0.75290404], [1.66730671, 0.78888889], [2.1281362 , 0.85265152], [2.5249616 , 0.9290404 ], [2.84178187, 0.9864899 ], [3.14260113, 1.02373737], [3.69303635, 1.06540404], [4.24987199, 1.10517677], [4.67549923, 1.13800505], [4.80030722, 1.13295455]]},
    # 6069, preflash
    {'serial':6069, 'binning':'1x1', 'date-after':'2020-03-28', 'airtemp_a':-2.0, 'airtemp_b':585.7, 'param1':[0.07433374, 0.79046048, 0.0810416 ,-0.25420098], 'param2':3.137078318644374,
     'points':[[1.00486431, 0.71944444], [1.37928827, 0.75290404], [1.66730671, 0.78888889], [2.1281362 , 0.85265152], [2.5249616 , 0.9290404 ], [2.84178187, 0.9864899 ], [3.14260113, 1.02373737], [3.69303635, 1.06540404], [4.24987199, 1.10517677], [4.67549923, 1.13800505], [4.80030722, 1.13295455]]},

    # 6132
    {'serial':6132, 'binning':'1x1', 'airtemp_a':0.516, 'airtemp_b':551.5, 'param1':[0.05483907, 0.84237167, 0.0629831 ,-0.18487527], 'param2':2.9551997417641154,
     'points':[[0.90565796, 0.79141414], [1.24487967, 0.80530303], [1.56810036, 0.82676768], [2.18894009, 0.9145202 ], [2.67537122, 0.97954545], [3.01139273, 1.00795455], [3.51062468, 1.03320707], [4.00985663, 1.06035354], [4.30107527, 1.08308081], [4.5858935 , 1.10265152], [4.76510497, 1.10012626]]},

    # 6149
    {'serial':6149, 'binning':'1x1', 'airtemp_a':0.85, 'airtemp_b':503.6, 'param1':[0.03364724, 0.90427548, 0.07500224,-0.21427316], 'param2':2.8578692493195548,
     'points':[[0.93445981, 0.80656566], [1.40809012, 0.83876263], [1.72491039, 0.86717172], [2.26574501, 0.93472222], [2.75217614, 0.9864899 ], [3.00819252, 1.00479798], [3.42101895, 1.01994949], [3.94585253, 1.03510101], [4.53149002, 1.05530303], [4.66909882, 1.0540404 ], [4.74910394, 1.04772727]]},

    # 6166 / SBT
    {'serial':6166, 'binning':'1x1', 'param1':[0.07392971, 0.77232407,-0.01752741, 0.01464123], 'param2':2.111340867505684,
     'points':[[1.00166411, 0.83939394], [1.53289811, 0.86338384], [1.78251408, 0.88042929], [2.28814644, 0.93409091], [2.76817716, 0.98712121], [3.07219662, 1.00795455], [3.47542243, 1.02941919], [3.82744496, 1.04393939], [4.19546851, 1.05782828], [4.51228879, 1.06792929], [4.69790067, 1.06792929], [4.81310804, 1.05972222]]},

    # 6167 / SBT
    {'serial':6167, 'binning':'1x1', 'param1':[0.10854498, 0.67574182, 0.02173521,-0.08850012], 'param2':2.6305813378315057,
     'point':[[0.96646185, 0.72449495], [1.53289811, 0.77121212], [1.67370712, 0.78825758], [2.33934972, 0.89810606], [2.74897593, 0.96881313], [3.01779314, 1.00416667], [3.54262673, 1.06098485], [4.04185868, 1.11022727], [4.41628264, 1.15063131], [4.5890937 , 1.1645202 ], [4.74270353, 1.16515152], [4.79710701, 1.15694444]]},

    # 6204 before readout upgrade
    {'serial':6204, 'binning':'1x1', 'date-before':'2019-12-01', 'airtemp_a':-1.55, 'airtemp_b':599.4, 'param1':[0.05399309, 0.84056592, 0.02539838,-0.06620873], 'param2':2.602603959616151,
     'points':[[1.38888889, 0.87790404], [1.80171531, 0.9145202 ], [2.30094726, 0.95555556], [2.68177163, 0.98333333], [2.88978495, 1.0010101 ], [3.41781874, 1.02373737], [3.83064516, 1.04393939], [4.23707117, 1.07613636], [4.5186892 , 1.0969697 ], [4.69470046, 1.10707071], [4.80350742, 1.09949495]]},
    {'serial':6204, 'binning':'2x2', 'date-before':'2019-12-01', 'airtemp_a':-1.51, 'airtemp_b':698.7},
    # 6204 after readout upgrade, seems nonlin did not change
    {'serial':6204, 'binning':'1x1', 'date-after':'2019-12-01', 'date-before':'2020-02-18', 'airtemp_a':-1.42, 'airtemp_b':574.1, 'param1':[0.05399309, 0.84056592, 0.02539838,-0.06620873], 'param2':2.602603959616151,
     'points':[[1.38888889, 0.87790404], [1.80171531, 0.9145202 ], [2.30094726, 0.95555556], [2.68177163, 0.98333333], [2.88978495, 1.0010101 ], [3.41781874, 1.02373737], [3.83064516, 1.04393939], [4.23707117, 1.07613636], [4.5186892 , 1.0969697 ], [4.69470046, 1.10707071], [4.80350742, 1.09949495]]},
    # 6204, preflash
    {'serial':6204, 'binning':'1x1', 'date-after':'2020-02-17', 'airtemp_a':-1.42, 'airtemp_b':574.1, 'param1':[0.05399309, 0.84056592, 0.02539838,-0.06620873], 'param2':2.602603959616151,
     'points':[[1.38888889, 0.87790404], [1.80171531, 0.9145202 ], [2.30094726, 0.95555556], [2.68177163, 0.98333333], [2.88978495, 1.0010101 ], [3.41781874, 1.02373737], [3.83064516, 1.04393939], [4.23707117, 1.07613636], [4.5186892 , 1.0969697 ], [4.69470046, 1.10707071], [4.80350742, 1.09949495]]},

    # 6205
    {'serial':6205, 'binning':'1x1', 'airtemp_a':-1.50, 'airtemp_b':584.0, 'param1':[0.04765044, 0.85458389,-0.03480524, 0.09953474], 'param2':2.8619181700630127,
     'points':[[0.97606247, 0.95113636], [1.70890937, 0.98017677], [2.15693804, 0.97954545], [2.58896569, 0.98522727], [3.09779826, 0.99974747], [3.72823861, 1.02563131], [4.21146953, 1.05782828], [4.57309268, 1.08371212], [4.70430108, 1.08813131], [4.80350742, 1.08055556]]},
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
        header = header.copy()

        if cfg and cfg.has_key('points'):
            points = np.array(cfg['points'])
            image = image.copy()
            idx = image > 1
            image[idx] /= interp1d(points[:,0], points[:,1], fill_value='extrapolate')(np.log10(image[idx]))

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
