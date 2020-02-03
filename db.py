#!/usr/bin/env python

import psycopg2, psycopg2.extras
import datetime

import numpy as np

class DB:
    """Class encapsulating the connection to PostgreSQL database"""
    def __init__(self, dbname='fram', dbhost='', dbport=0, dbuser='', dbpassword='', readonly=False):
        connstring = "dbname=" + dbname
        if dbhost:
            connstring += " host="+dbhost
        if dbport:
            connstring += " port=%d" % dbport
        if dbuser:
            connstring += " user="+dbuser
        if dbpassword:
            connstring += " password='%s'" % dbpassword

        self.connect(connstring, readonly)

    def connect(self, connstring, readonly=False):
        self.conn = psycopg2.connect(connstring)
        self.conn.autocommit = True
        self.conn.set_session(readonly=readonly)
        psycopg2.extras.register_default_jsonb(self.conn)
        # FIXME: the following adapter is registered globally!
        psycopg2.extensions.register_adapter(dict, psycopg2.extras.Json)

        self.connstring = connstring
        self.readonly = readonly

    def query(self, string="", data=(), simplify=True, debug=False, array=False):
        if self.conn.closed:
            print "Re-connecting to DB"
            self.connect(self.connstring, self.readonly)

        cur = self.conn.cursor(cursor_factory = psycopg2.extras.DictCursor)

        if debug:
            print cur.mogrify(string, data)

        if data:
            cur.execute(string, data)
        else:
            cur.execute(string)

        try:
            result = cur.fetchall()
            # Simplify the result if it is simple
            if array:
                # Code from astrolibpy, https://code.google.com/p/astrolibpy
                strLength = 10
                __pgTypeHash = {
                    16:bool,18:str,20:'i8',21:'i2',23:'i4',25:'|S%d'%strLength,700:'f4',701:'f8',
                    1042:'|S%d'%strLength,#character()
                    1043:'|S%d'%strLength,#varchar
                    1114:'|O',#datetime
                    1700:'f8' #numeric
                }

                desc = cur.description
                names = [d.name for d in desc]
                formats = [__pgTypeHash.get(d.type_code, '|O') for d in desc]

                table = np.recarray(shape=(cur.rowcount,), formats=formats, names=names)

                for i,v in enumerate(result):
                    table[i] = tuple(v)

                return table
            elif simplify and len(result) == 1:
                if len(result[0]) == 1:
                    return result[0][0]
                else:
                    return result[0]
            else:
                return result
        except:
            # Nothing returned from the query
            #import traceback
            #traceback.print_exc()
            return None

    def get_stars(self, ra0=0, dec0=0, sr0=0, limit=10000, catalog='pickles', extra=[], extrafields=None):
        # Code from astrolibpy, https://code.google.com/p/astrolibpy
        strLength = 10
        __pgTypeHash = {
            16:bool,18:str,20:'i8',21:'i2',23:'i4',25:'|S%d'%strLength,700:'f4',701:'f8',
            1042:'|S%d'%strLength,#character()
            1043:'|S%d'%strLength,#varchar
            1114:'|O',#datetime
            1700:'f8' #numeric
        }

        substr = ""

        if catalog == 'tycho2':
            substr = "0.76*bt+0.24*vt as b , 1.09*vt-0.09*bt as v, 0 as r"

        # TODO: Do we really need brightess ordered output?..
        order = ""
        if False:
            if catalog in ['tycho2', 'apass', 'pickles']:
                order = "ORDER BY v"
            elif catalog in ['twomass']:
                order = "ORDER BY j"
            elif catalog in ['atlas']:
                order = "ORDER BY g"

        if extra and type(extra) == list:
            extra_str = " AND " + " AND ".join(extra)
        elif extra and type(extra) == str:
            extra_str = " AND " + extra
        else:
            extra_str = ""

        if extrafields:
            if substr:
                substr = substr + "," + extrafields
            else:
                substr = extrafields

        if substr:
            substr = "," + substr

        cur = self.conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
        cur.execute("SELECT * " + substr + " FROM " + catalog + " cat WHERE q3c_radial_query(ra, dec, %s, %s, %s) " + extra_str + " " + order + " LIMIT %s;", (ra0, dec0, sr0, limit))

        desc = cur.description
        names = [d.name for d in desc]
        formats = [__pgTypeHash.get(d.type_code, '|O') for d in desc]

        table = np.recarray(shape=(cur.rowcount,), formats=formats, names=names)

        for i,v in enumerate(cur.fetchall()):
            table[i] = tuple(v)

        # Add some computed fields to the table
        if catalog == 'pickles':
            # Pickles - has computed B, V, R and measured J, H, K, Bt, Vt
            table = np.lib.recfunctions.append_fields(table,
                        ['B', 'V', 'R', 'I', 'Berr', 'Verr', 'Rerr', 'Ierr'],
                        [
                            # B
                            0.760*table['bt'] + 0.240*table['vt'], # B = 0.76*BT+0.24*VT
                            # V
                            table['vt'] - 0.090*(table['bt'] - table['vt']), # V = VT -0.090*(BT-VT)
                            # R
                            table['r'],
                            # I
                            table['vt'] - 0.090*(table['bt'] - table['vt']) - 1.6069*(table['j'] - table['k']) + 0.0503, # V - Ic = 1.6069 * (J - Ks) + 0.0503
                            # Berr
                            np.hypot(table['ebt'], table['evt']),
                            # Verr
                            np.hypot(table['ebt'], table['evt']),
                            # Rerr
                            np.hypot(table['ebt'], table['evt']),
                            # Ierr
                            np.hypot(table['ebt'], table['evt']),
                        ],
                        [np.double, np.double, np.double, np.double, np.double, np.double, np.double, np.double])

        elif catalog == 'apass':
            # APASS - has measured B, V, g, r, i
            table = np.lib.recfunctions.append_fields(table,
                        ['B', 'V', 'R', 'I', 'Berr', 'Verr', 'Rerr', 'Ierr'],
                        [
                            # B
                            table['b'],
                            # V
                            table['v'],
                            # R
                            table['r'] - 0.153*(table['r'] - table['i']) - 0.117, # R-r = (-0.153 +/- 0.003)*(r-i) - (0.117 +/- 0.003)
                            # I
                            table['v'] - 0.675*(table['g'] - table['i']) - 0.364, # V-I = (0.675 +/- 0.002)*(g-i)  + (0.364 +/- 0.002)
                            # Berr
                            table['berr'],
                            # Verr
                            table['verr'],
                            # Rerr
                            table['rerr'],
                            # Ierr
                            table['ierr'],
                        ],
                        [np.double, np.double, np.double, np.double, np.double, np.double, np.double, np.double])

        elif catalog == 'atlas':
            # ATLAS-refcat2 - has measured Gaia, GaiaBP, GaiaRP, g, r, i, z
            table = np.lib.recfunctions.append_fields(table,
                        ['B', 'V', 'R', 'I', 'Berr', 'Verr', 'Rerr', 'Ierr'],
                        [
                            # B
                            table['g'] + 0.313*(table['g'] - table['r']) + 0.219, # B-g = (0.313 +/- 0.003)*(g-r)  + (0.219 +/- 0.002)
                            # V
                            table['g'] - 0.565*(table['g'] - table['r']) - 0.016, # V-g = (-0.565 +/- 0.001)*(g-r) - (0.016 +/- 0.001)
                            # R
                            table['r'] - 0.153*(table['r'] - table['i']) - 0.117, # R-r = (-0.153 +/- 0.003)*(r-i) - (0.117 +/- 0.003)
                            # I
                            table['i'] - 0.386*(table['i'] - table['z']) - 0.397, # I-i = (-0.386 +/- 0.004)*(i-z) - (0.397 +/- 0.001)
                            # Berr
                            table['dg'],
                            # Verr
                            table['dg'],
                            # Rerr
                            table['dr'],
                            # err
                            table['di'],
                        ],
                        [np.double, np.double, np.double, np.double, np.double, np.double, np.double, np.double])

        return table
