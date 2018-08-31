#!/usr/bin/env python

__DAY__ = 3

import util

def is_valid(a,b,c):
    return a+b>c and a+c>b and b+c>a

def solve(data):
    valid_horizontally = 0
    valid_vertically = 0
    sides = map(int, data.split())
    for triangle in [sides[n:n+3] for n in range(0,len(sides),3)]:
        valid_horizontally += is_valid(*triangle)
    for triangle in [sides[n+i:n+9:3] for n in range(0,len(sides),9) for i in range(3)]:
        valid_vertically += is_valid(*triangle)

    print("Part 1: {0} triangles are possible".format(valid_horizontally))
    print("Part 2: {0} triangles are possible".format(valid_vertically))

if __name__ == '__main__':
    data = util.get_input(__DAY__)
    solve(data)
