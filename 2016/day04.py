#!/usr/bin/env python

__DAY__ = 4

from itertools import groupby
import re
from string import ascii_lowercase, maketrans

import util

room_pattern = re.compile(r'^(?P<name>(?:[a-z]+-)+)(?P<sector_id>\d+)\[(?P<checksum>[a-z]+)\]$', re.MULTILINE)

def solve(data):
    total = 0
    storage_sector = None

    for m in room_pattern.finditer(data):
        name = m.group('name')
        sector_id = int(m.group('sector_id'))
        letter_groups = ((k, list(v)) for k,v in groupby(sorted(name.replace('-', ''))))
        letters = sorted(letter_groups, key=lambda k, v: len(v), reverse=True)
        checksum = ''.join(k for (k,v) in letters)[:5]
        total += sector_id if checksum == m.group('checksum') else 0

        key = ascii_lowercase[sector_id%26:] + ascii_lowercase[:sector_id%26]
        table = maketrans(ascii_lowercase, key)
        if name.translate(table).replace('-', ' ').strip() == 'northpole object storage':
            storage_sector = sector_id
    
    print("Part 1: Sum of the Sector IDs is: {0}".format(total))
    print("Part 2: Sector ID of where North Pole objects is: {0}".format(storage_sector))

if __name__ == '__main__':
    data = util.get_input(__DAY__)
    solve(data)
