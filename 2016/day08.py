#!/usr/bin/env python

__DAY__ = 8

from collections import deque
import re

import util

line_pattern = re.compile('''\
    (?P<rect>rect\s(?P<width>\d+)x(?P<height>\d+))
|   (rotate\s(?P<rot_type>row|column)\s[xy]=(?P<num>\d+)\sby\s(?P<shift>\d+))
''', re.VERBOSE)

class G(object):
    def __init__(self, height, width):
        self.g = []
        for row in range(height):
            self.g.append(['.' for col in range(width)])

    def rect(self, a,b):
        for row in range(b):
            for col in range(a):
                self.g[row][col] = '#'

    def rotate_col(self, a, b):
        new_col = deque(row[a] for row in self.g)
        new_col.rotate(b)
        for row, cell in zip(self.g, new_col):
            row[a] = cell

    def rotate_row(self, a, b):
        new_row = deque(self.g[a])
        new_row.rotate(b)
        self.g[a] = list(new_row)
    
    def solutions(self):
        code = '\n'.join(''.join(row) for row in self.g)
        lit = code.count('#')
        return lit, code

def solve(data):
    g = G(height=6, width=50)
    for m in line_pattern.finditer(data):
        if m.group('rect'):
            g.rect(int(m.group('width')), int(m.group('height')))
        elif m.group('rot_type') == 'row':
            g.rotate_row(int(m.group('num')), int(m.group('shift')))
        else:
            g.rotate_col(int(m.group('num')), int(m.group('shift')))
    lit, code = g.solutions()
    print("Part 1: Lights lit: {0}".format(lit))
    print("Part 2: Code:\n{0}".format(code))

if __name__ == '__main__':
    data = util.get_input(__DAY__)
    solve(data)
