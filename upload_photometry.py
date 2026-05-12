#!/usr/bin/env python

import os
import re
import sys
import datetime

import numpy as np

from fram.fram import Fram
from astropy.table import Table

from io import StringIO


COPY_COLUMNS = [
    "image",
    "time",
    "filter",
    "ra",
    "dec",
    "mag",
    "magerr",
    "flags",
    "mag_color",
    "color_term",
    "std",
    "zp_std",
    "nstars",
    "final_frac",
    "fwhm",
]


def touch(filename):
    with open(filename, "a"):
        pass

    os.utime(filename, None)


def validate_table_name(table):
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
        raise ValueError("Invalid table name: %s" % table)

    return table


def load_objects(filename):
    if not os.path.getsize(filename):
        return None

    return Table.read(filename, format="parquet")


def get_meta(obj, name, default=None):
    value = obj.meta.get(name, default)

    if np.ma.is_masked(value):
        return default

    return value


def format_time(value):
    if value is None:
        return None

    if hasattr(value, "to_datetime"):
        value = value.to_datetime()
    elif hasattr(value, "datetime"):
        value = value.datetime

    if isinstance(value, datetime.datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S.%f")

    return value


def copy_value(value):
    if value is None or np.ma.is_masked(value):
        return r"\N"

    if isinstance(value, np.generic):
        value = value.item()

    return str(value)


def append_object_rows(buf, obj, image_id):
    time = format_time(get_meta(obj, "time"))
    filter_name = get_meta(obj, "filter", get_meta(obj, "filter_name"))
    color_term = get_meta(obj, "color_term")
    std = get_meta(obj, "std")
    zp_std = np.std(obj['zp'])
    nstars = get_meta(obj, "nstars")
    final_frac = get_meta(obj, "final_frac")

    for i in range(len(obj["ra"])):
        values = [
            image_id,
            time,
            filter_name,
            obj["ra"][i],
            obj["dec"][i],
            obj["mag_calib"][i],
            obj["mag_calib_err"][i],
            int(obj["flags"][i]),
            obj["mag_calib_color"][i],
            color_term,
            std,
            zp_std,
            nstars,
            final_frac,
            obj["fwhm"][i],
        ]
        print(*[copy_value(_) for _ in values], sep="\t", file=buf)


def flush(cur, conn, buf, table, filenames):
    buf.seek(0)
    cur.copy_from(buf, table, sep="\t", null=r"\N", columns=COPY_COLUMNS, size=65535000)

    print("committing...")
    conn.commit()

    for fn in filenames:
        if not os.path.exists(fn + ".upload"):
            touch(fn + ".upload")


if __name__ == "__main__":
    from optparse import OptionParser

    import socket

    if socket.gethostname() == "s122":
        host = "s104.fzu.cz"
    else:
        host = None

    parser = OptionParser(usage="usage: %prog [options] arg")
    parser.add_option(
        "-d",
        "--db",
        help="Database name",
        action="store",
        dest="db",
        type="str",
        default="fram",
    )
    parser.add_option(
        "-H",
        "--host",
        help="Database host",
        action="store",
        dest="dbhost",
        type="str",
        default=host,
    )
    parser.add_option(
        "-t",
        "--table",
        help="Database table",
        action="store",
        dest="table",
        type="str",
        default="photometry",
    )
    parser.add_option(
        "-c",
        "--chunk",
        help="Commit chunk size",
        action="store",
        dest="chunk",
        type="int",
        default=500,
    )
    parser.add_option(
        "-r",
        "--replace",
        help="Replace already existing records in database",
        action="store_true",
        dest="replace",
        default=False,
    )
    parser.add_option(
        "-i",
        "--ignore",
        help="Ignore upload status file",
        action="store_true",
        dest="ignore",
        default=False,
    )
    parser.add_option(
        "-v",
        "--verbose",
        help="Verbose",
        action="store_true",
        dest="verbose",
        default=False,
    )

    (options, files) = parser.parse_args()

    fram = Fram(dbname=options.db, dbhost=options.dbhost)
    fram.conn.autocommit = False
    cur = fram.conn.cursor()

    N = 0

    s = StringIO()
    filenames = []

    table = validate_table_name(options.table)

    for i, filename in enumerate(files):
        if (
            not options.replace
            and not options.ignore
            and os.path.exists(filename + ".upload")
        ):
            # print('Skipping', filename, 'as upload file exists')
            continue

        obj = load_objects(filename)

        if obj is None:
            if not options.replace:
                touch(filename + ".upload")
            continue

        image_filename = get_meta(obj, "filename")

        if image_filename is None:
            print(
                "Skipping",
                filename,
                "as source image filename is absent from metadata",
                file=sys.stderr,
            )
            continue

        image_id = fram.query(
            "SELECT id FROM images WHERE filename=%s",
            (image_filename,),
            simplify=True,
            table=False,
        )

        if image_id is None:
            print(
                "Skipping",
                filename,
                "as image is absent from DB:",
                image_filename,
                file=sys.stderr,
            )
            continue

        if False: # slow
            if options.replace:
                fram.query("DELETE FROM " + table + " WHERE image=%s", (image_id,))

            elif fram.query(
                "SELECT EXISTS(SELECT 1 FROM " + table + " WHERE image=%s);",
                (image_id,),
                simplify=True,
                table=False,
            ):
                if not options.replace:
                    touch(filename + ".upload")

                # print('Skipping', filename, 'as already in DB')
                continue

        if options.verbose:
            print(image_filename, image_id)

        if len(files) > 1:
            print(i, "/", len(files), filename, len(obj["ra"]))
            sys.stdout.flush()

        append_object_rows(s, obj, image_id)
        filenames.append(filename)

        N += 1

        if N % options.chunk == 0:
            flush(cur, fram.conn, s, table, filenames)
            s = StringIO()
            filenames = []

    if filenames:
        flush(cur, fram.conn, s, table, filenames)

    print(N, "/", len(files), "files uploaded")
