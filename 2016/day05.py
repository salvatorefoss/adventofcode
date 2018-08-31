#!/usr/bin/env python

__DAY__ = 5

from hashlib import md5
from itertools import count, islice

import util

def interesting_hash_generator(password):
    for n in count():
        h = md5(password + str(n)).hexdigest()
        if h.startswith('00000'):
            yield h[5], h[6]

def solve(data):
    password = ''.join(k for k,v in islice(interesting_hash_generator(data), 8))
    password_2 = {
        '0': None,
        '1': None,
        '2': None,
        '3': None,
        '4': None,
        '5': None,
        '6': None,
        '7': None
    }
    g = interesting_hash_generator(data)
    while not all(password_2.values()):
        k, v = next(g)
        if k in password_2 and not password_2[k]:
            password_2[k] = v 
    password_2 = ''.join(v for k,v in sorted(password_2.items()))
    print("Part 1: Password is: {0}".format(password))
    print("Part 2: Second password is: {0}".format(password_2))

if __name__ == '__main__':
    data = util.get_input(__DAY__)
    solve(data)
