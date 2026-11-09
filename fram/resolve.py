from __future__ import absolute_import, division, print_function, unicode_literals

import xml.dom.minidom as minidom
import xml.parsers.expat
import re

import requests

# Sesame servers tried, in order, when the previous one is unreachable, answers
# with an HTTP error, or answers with something that is not XML (an outage page)
sesame_mirrors = [
    'https://cds.unistra.fr/cgi-bin/nph-sesame',
    'https://vizier.cfa.harvard.edu/viz-bin/nph-sesame',
]

# Short connect timeout, so that a server which does not accept connections at
# all (the usual way CDS goes down) is skipped quickly; a longer read timeout,
# as NED is slow to answer on a miss
sesame_timeout = (5, 30)

def simbadResolve(name = 'm31'):
    doc = None

    for server in sesame_mirrors:
        try:
            res = requests.get(server + '/-oxpi/SN',
                               params=requests.utils.quote(name),
                               timeout=sesame_timeout)
            res.raise_for_status()
            doc = minidom.parseString(res.content)
            break
        except (requests.exceptions.RequestException, xml.parsers.expat.ExpatError):
            continue

    if doc is None:
        # No server could be asked
        return "", 0, 0

    try:
        r = doc.getElementsByTagName('Resolver')[0]

        name = r.getElementsByTagName('oname')[0].childNodes[0].nodeValue
        ra = float(r.getElementsByTagName('jradeg')[0].childNodes[0].nodeValue)
        dec = float(r.getElementsByTagName('jdedeg')[0].childNodes[0].nodeValue)

        return name, ra, dec
    except:
        return "", 0, 0

def parseSexadecimal(string):
    value = 0

    m = (re.search("^\s*([+-])?\s*(\d{1,3})\s+(\d{1,2})\s+(\d{1,2}\.?\d*)\s*$", string) or
         re.search("^\s*([+-])?\s*(\d{1,3})\:(\d{1,2})\:(\d{1,2}\.?\d*)\s*$", string))
    if m:
        value = float(m.group(2)) + float(m.group(3))/60 + float(m.group(4))/3600

        if m.group(1) == '-':
            value = -value

    return value

def resolve(string = ''):
    """
    Resolve the object name (or coordinates string) into proper coordinates on the sky
    """
    name = ''
    ra = 0
    dec = 0

    m = re.search("^\s*(\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s*$", string)
    if m:
        name = 'degrees'
        ra = float(m.group(1))
        dec = float(m.group(2))
    else:
        m = (re.search("^\s*(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}\.?\d*)\s+([+-])?\s*(\d{1,3})\s+(\d{1,2})\s+(\d{1,2}\.?\d*)\s*$", string) or
             re.search("^\s*(\d{1,2})\:(\d{1,2})\:(\d{1,2}\.?\d*)\s+([+-])?\s*(\d{1,3})\:(\d{1,2})\:(\d{1,2}\.?\d*)\s*$", string))
        if m:
            name = 'sexadecimal'
            ra = (float(m.group(1)) + float(m.group(2))/60 + float(m.group(3))/3600)*15
            dec = (float(m.group(5)) + float(m.group(6))/60 + float(m.group(7))/3600)

            if m.group(4) == '-':
                dec = -dec
        else:
            name, ra, dec = simbadResolve(string)

    return name, ra, dec
