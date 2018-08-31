#!/usr/bin/env python

__DAY__ = 2

import util

class Codepad(object):
    def __init__(self):
        self.basic_codepad = (
            'XXXXX',
            'X123X',
            'X456X',
            'X789X',
            'XXXXX'
        )

        self.fancy_codepad = (
            'XXXXXXX',
            'XXX1XXX',
            'XX234XX',
            'X56789X',
            'XXABCXX',
            'XXXDXXX',
            'XXXXXXX'
        )
        self.basic_coords = (2, 2)
        self.basic_code = []
        self.fancy_coords = (1, 3)
        self.fancy_code = []

    def step(self, direction):
        self.basic_coords = self.step_codepad(self.basic_coords, self.basic_codepad, direction)
        self.fancy_coords = self.step_codepad(self.fancy_coords, self.fancy_codepad, direction)

    def step_codepad(self, from_coords, codepad, direction):
        new_coords = {
            'U': (from_coords[0], from_coords[1] - 1),
            'R': (from_coords[0] + 1, from_coords[1]),
            'D': (from_coords[0], from_coords[1] + 1),
            'L': (from_coords[0] - 1, from_coords[1])
        }[direction]
        if not codepad[new_coords[1]][new_coords[0]] == 'X':
            return new_coords
        return from_coords

    def press(self):
        self.basic_code.append(self.basic_codepad[self.basic_coords[1]][self.basic_coords[0]])
        self.fancy_code.append(self.fancy_codepad[self.fancy_coords[1]][self.fancy_coords[0]])

    def solutions(self):
        return ''.join(map(str, self.basic_code)), ''.join(self.fancy_code)

def solve(data):
    codepad = Codepad()
    for line in data.splitlines():
        for direction in line:
            codepad.step(direction)
        codepad.press()
    basic_code, fancy_code = codepad.solutions()
    print("Part 1: Basic code is {0}".format(basic_code))
    print("Part 2: Fancy code is {0}".format(fancy_code))

if __name__ == '__main__':
    data = util.get_input(__DAY__)
    solve(data)
