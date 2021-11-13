from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import datetime, os, sys

from esutil import coords, htm
import statsmodels.api as sm
from scipy.spatial import cKDTree

from supersmoother import SuperSmoother

from . import survey, calibrate

class LCs:
    def __init__(self):
        self.fid = 0

        self.fids = []
        self.filenames = []

        self.times, self.filters = [],[]
        self.ras, self.decs = [],[]
        self.xs, self.ys = [],[]
        self.mags, self.magerrs, self.flags = [],[],[]
        self.stds, self.nstars = [],[]
        self.corrs = []
        self.dmags = []
        self.omags = []
        self.bgs = []

        self.lcs = []

    def add(self, time, ra, dec, mag, magerr=None, flags=None, filter=None, std=None, nstars=None, x=None, y=None, corr=None, filename=None, dmag=0.0, omag=None, bg=None):
        def extend(col, val):
            if val is not None and hasattr(val, "__len__") and len(val) == len(ra):
                col.extend(val)
            elif val is not None:
                col.extend(np.repeat(val, len(ra)))
            else:
                col.extend(np.repeat(None, len(ra)))

        extend(self.times, time)
        # extend(self.filenames, filename)

        self.filenames.append(filename)

        extend(self.filters, filter)
        extend(self.ras, ra)
        extend(self.decs, dec)
        extend(self.mags, mag)
        extend(self.magerrs, magerr)
        extend(self.flags, flags)
        extend(self.xs, x)
        extend(self.ys, y)
        extend(self.stds, std)
        extend(self.nstars, nstars)
        extend(self.corrs, corr)
        extend(self.dmags, dmag)
        extend(self.omags, omag)
        extend(self.bgs, bg)

        extend(self.fids, self.fid)
        self.fid += 1

    def cluster(self, sr=15/3600, min_length = None, verbose=True):
        if min_length is None:
            min_length = int(0.7*len(np.unique(self.fids)))

        sr0 = np.deg2rad(sr)

        if type(self.times) is not np.ndarray:
            if verbose:
                print('Converting arrays')
            self.fids, self.filenames, self.times, self.filters, self.ras, self.decs, self.xs, self.ys, self.mags, self.magerrs, self.flags, self.stds, self.nstars, self.corrs, self.dmags, self.omags, self.bgs = [np.array(_) for _ in self.fids, self.filenames, self.times, self.filters, self.ras, self.decs, self.xs, self.ys, self.mags, self.magerrs, self.flags, self.stds, self.nstars, self.corrs, self.dmags, self.omags, self.bgs]

            self.xarr,self.yarr,self.zarr = survey.radectoxyz(self.ras, self.decs)
            self.kd = cKDTree(np.array([self.xarr, self.yarr, self.zarr]).T)

        def refine_pos(x, y, z):
            x1,y1,z1 = [np.mean(_) for _ in x,y,z]
#             dx1,dy1,dz1 = [1.4826*np.median(np.abs(_)) for _ in x-x1, y-y1, z-z1]
            r = np.sqrt(x1*x1 + y1*y1 + z1*z1)

            x1,y1,z1 = [_/r for _ in x1,y1,z1]
#             dx1,dy1,dz1 = [_/r for _ in dx1,dy1,dz1]

            dr = 0 # 2.0*np.sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)

            return x1,y1,z1,dr

        vmask = np.zeros_like(self.times, np.bool)
        vmask[(self.flags > 0)] = True # ???

        if verbose:
            print(len(vmask), 'photometric points,', len(vmask[vmask]), 'pre-masked')

        mean,median,std,rstd,std,rstd0,std0,err,chi2 = [],[],[],[],[],[],[],[],[]
        xs,ys,zs = [],[],[]
        Ns = []

        N0 = 0

        mags = self.mags + self.dmags

        for i in xrange(len(vmask)):
            if not vmask[i]:
                ids = self.kd.query_ball_point([self.xarr[i], self.yarr[i], self.zarr[i]], sr0)

                if len(ids) < min_length:
                    vmask[ids] = True
                else:

                    x1,y1,z1,dr1 = refine_pos(self.xarr[ids], self.yarr[ids], self.zarr[ids])
                    dr1 = sr0
                    ids = self.kd.query_ball_point([x1,y1,z1], max(dr1, sr0))
                    vmask[ids] = True
#                     ids = self.kd.query_ball_point([x1,y1,z1], dr1)

                    if len(ids) >= min_length:
                        idx = self.flags[ids] == 0

                        if np.sum(idx) < min_length:
                            continue

                        v = mags[ids][idx]
                        v0 = self.mags[ids][idx]
                        verr = self.magerrs[ids][idx]

                        ss = self.stds[ids][idx]
                        idx2 = ss < np.median(ss) + 3.0*calibrate.rstd(ss)

                        if np.sum(idx2) < min_length:
                            continue

                        mean.append(np.mean(v[idx2]))
                        median.append(np.median(v[idx2]))
                        std.append(np.std(v[idx2]))
                        std0.append(np.std(v0[idx2]))
                        rstd.append(calibrate.rstd(v[idx2]))
                        rstd0.append(calibrate.rstd(v0[idx2]))
                        err.append(np.mean(verr[idx2]))

                        xs.append(x1)
                        ys.append(y1)
                        zs.append(z1)
                        Ns.append(np.sum(idx))
                        N0 += 1

                    if i % 100 == 0 and verbose:
                        sys.stdout.write("\r %d points - %d lcs" % (i, N0))
                        sys.stdout.flush()

        if len(xs):
            kds = cKDTree(np.array([xs,ys,zs]).T)
        mean,median,std,std0,rstd,rstd0,err,xs,ys,zs,Ns = [np.array(_) for _ in mean,median,std,std0,rstd,rstd0,err,xs,ys,zs,Ns]
        ra,dec = survey.xyztoradec([xs, ys, zs])

        if verbose:
            print()
            print(len(mean), 'lcs with more than', min_length, 'points isolated')

        # Mark the variability
        idx = np.ones_like(mean, dtype=np.bool)
        model = SuperSmoother()

        for i in xrange(5):
            try:
                model.fit(mean[idx], std[idx], std[idx])
                p_std = model.predict(mean)
            except:
                p = np.polyfit(mean[idx], std[idx], 5, w=1.0/std[idx])
                p_std = np.polyval(p, mean)

            pm,ps = np.mean((std/p_std)[idx]),np.std((std/p_std)[idx])
            idx = np.abs(std/p_std - pm) < 4.0*ps

        var = np.abs(std/p_std - pm) > 4.0*ps

        self.lcs = {'mean': mean, 'median': median, 'std': std, 'std0': std0, 'rstd': rstd, 'rstd0': rstd0, 'err': err, 'ra': ra, 'dec': dec, 'N': Ns, 'var': var, 'kd':kds}
