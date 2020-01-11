from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from scipy.stats import sigmaclip

# Robust mean
def rmean(data):
    return np.mean(sigmaclip(data, 3.0, 3.0)[0])

def rstd(data):
    return np.median(np.abs(data - np.median(data)))

def parse_det(string):
    x0,x1,y0,y1 = [int(_)-1 for _ in sum([_.split(':') for _ in string[1:-1].split(',')], [])]

    return x0,x1,y0,y1

# Cropping of overscans if any
def crop_overscans(image, header=None, subtract=True):
    if header is not None:
        header = header.copy()

    if subtract:
        # Subtract bias region
        bias = None

        if image.shape == (4124, 4148): # Initial patched G4 firmware
            bias = rmean(list(image[2:8, 300:-300].flatten()) + list(image[-14:, 300:-300].flatten()))

        elif image.shape == (4127,4144): # Official firmwares after enabling overscans in Windows utility
            bias = rmean(list(image[3:7, 300:-300].flatten()) + list(image[-17:-2, 300:-300].flatten()))

        elif image.shape == (1026, 1062): # Overscan-enabled custom G2 on La Palma
            bias = rmean(list(image[:, -5:]))

        if bias is not None:
            image = image.copy() - bias

            if header is not None:
                header['BIASAVG'] = bias

    if header is None or not header.get('DATASEC'):
        # Special handling of legacy G4 data - manually adjusted overscan-free regions
        if image.shape == (4124, 4148): # Initial patched G4 firmware
            image = image[11:-17, 33:-19]

            if header is not None and header.get('CRPIX1') is not None:
                header['CRPIX1'] -= 33
                header['CRPIX2'] -= 11
            if header is not None:
                header['DATASEC'] = '[34:4129,12:4107]'

        elif image.shape == (4127,4144): # Official firmwares after enabling overscans in Windows utility
            image = image[11:-20, 30:-18]

            if header is not None and header.get('CRPIX1') is not None:
                header['CRPIX1'] -= 30
                header['CRPIX2'] -= 11
            if header is not None:
                header['DATASEC'] = '[31:4126,12:4107]'

        elif image.shape == (1026, 1062): # Overscan-enabled custom G2 on La Palma
            image = image[2:, 0:1056]

            if header is not None and header.get('CRPIX1') is not None:
                header['CRPIX1'] -= 0
                header['CRPIX2'] -= 2
            if header is not None:
                header['DATASEC'] = '[1:1056,3:1026]'

    else:
        x1,x2,y1,y2 = parse_det(header.get('DATASEC'))

        if header is not None and header.get('CRPIX1') is not None:
            header['CRPIX1'] -= x1
            header['CRPIX2'] -= y1

        image = image[y1:y2+1, x1:x2+1]

    image = np.ascontiguousarray(image)

    if header is not None:
        header['NAXIS1'] = image.shape[1]
        header['NAXIS2'] = image.shape[0]

        return image,header
    else:
        return image

def calibrate(image, header, dark=None, crop=True, subtract=True, linearize=True):
    if crop:
        image,header = crop_overscans(image, header, subtract=subtract)

    if dark is not None:
        if dark.shape != image.shape:
            print("Wrong dark shape:", dark.shape, "vs", image.shape)
        else:
            image = 1.0*image - dark

    if linearize:
        serial = header['product_id']

        if serial == 6029:
            if header['DATE'] < '2019-07-12':
                # Darkroom / 06029 - FRAM WF7
                param1, param2 = [ 0.07290866, 0.81854223, 0.1754477 ,-0.55883105], 3.173551818062375
            else:
                # New Darkroom / 06029
                param1, param2 = [ 0.05545345, 0.86168363, 0.14405999,-0.45671358], 3.1756676670297868
        elif serial == 6069:
            # Darkroom / 06069 - CTA-S0 FRAM
            param1, param2 = [ 0.07433374, 0.79046048, 0.0810416 ,-0.25420098], 3.137078318644374
        elif serial == 6132:
            # Darkroom / 06132 - CTA-S1 FRAM
            param1, param2 = [ 0.05483907, 0.84237167, 0.0629831 ,-0.18487527], 2.9551997417641154
        elif serial == 6149:
            # Darkroom / 06149 - CTA-N FRAM
            param1, param2 = [ 0.03364724, 0.90427548, 0.07500224,-0.21427316], 2.8578692493195548
        elif serial == 6166:
            # Darkroom / 06166 - BART SBT
            param1, param2 = [ 0.07392971, 0.77232407,-0.01752741, 0.01464123], 2.111340867505684
        elif serial == 6167:
            # Darkroom / 06167 - BART SBT
            param1, param2 = [ 0.10854498, 0.67574182, 0.02173521,-0.08850012], 2.6305813378315057
        elif serial == 6204:
            if header['DATE'] < '2019-11-30':
                # Darkroom / 06204 - FRAM WF8
                param1, param2 = [ 0.05861087, 0.82682591, 0.04755189,-0.09645298], 2.0240170211992154
            else:
                # New Darkroom / 06204
                param1, param2 = [ 0.05399309, 0.84056592, 0.02539838,-0.06620873], 2.602603959616151
        elif serial == 6205:
            # Darkroom / 06205 - FRAM NF4
            param1, param2 = [ 0.04765044, 0.85458389,-0.03480524, 0.09953474], 2.8619181700630127

        elif serial == 2596:
            # La Palma custom G2:
            param1, param2 = [-0.00908378, 1.01761044, 0, 0], 0

        else:
            print("Unsupported chip", serial)
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
