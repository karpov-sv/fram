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

    def get_stars(self, ra0=0, dec0=0, sr0=0, limit=10000, catalog='pickles', as_list=False, extra=[], extrafields=None):
        # Code from astrolibpy, https://code.google.com/p/astrolibpy
        strLength = 10
        __pgTypeHash = {
            16:bool,18:str,20:'i8',21:'i2',23:'i4',25:'|S%d'%strLength,700:'f4',701:'f8',
            1042:'|S%d'%strLength,#character()
            1043:'|S%d'%strLength,#varchar
            1114:'|O',#datetime
            1700:'f8' #numeric
        }

        substr, order = "", "v"
        if catalog == 'tycho2':
            substr = "0.76*bt+0.24*vt as b , 1.09*vt-0.09*bt as v, 0 as r"
        elif catalog == 'twomass':
            order = "j"

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
        cur.execute("SELECT * " + substr + " FROM " + catalog + " cat WHERE q3c_radial_query(ra, dec, %s, %s, %s) " + extra_str + " ORDER BY " + order + " ASC LIMIT %s;", (ra0, dec0, sr0, limit))

        if as_list:
            return cur.fetchall()
        else:
            desc = cur.description
            names = [d.name for d in desc]
            formats = [__pgTypeHash.get(d.type_code, '|O') for d in desc]

            table = np.recarray(shape=(cur.rowcount,), formats=formats, names=names)

            for i,v in enumerate(cur.fetchall()):
                table[i] = tuple(v)

            return table
