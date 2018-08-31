#!/usr/bin/env python

__DAY__ = 1

import re

import util

seq_pattern = re.compile("(?P<direction>[RL])(?P<blocks>\d+)")

class Sleigh(object):
    def __init__(self):
        self.x = 0
        self.y = 0

        self.facing = 0

        self.hq = None
        self.locations_visited = set()

    def move(self, direction, blocks):
        self.facing += 1 if direction == 'R' else -1
        
        for n in range(int(blocks)):
            self.step()
            if not self.hq and (self.x, self.y) in self.locations_visited:
                self.hq = (self.x, self.y)
            self.locations_visited.add((self.x, self.y))

    def step(self):
        self.x, self.y = {
            0: (self.x, self.y + 1),
            1: (self.x + 1, self.y),
            2: (self.x, self.y - 1),
            3: (self.x - 1, self.y)
        }[self.facing % 4]

    def solutions(self):
        return abs(self.x) + abs(self.y), abs(self.hq[0]) + abs(self.hq[1])

def solve(data):
    sleigh = Sleigh()
    for m in seq_pattern.finditer(data):
        sleigh.move(**m.groupdict())
    destination_distance, hq_distance = sleigh.solutions()
    print("Part 1: Bunny HQ is {0} blocks away".format(destination_distance))
    print("Part 2: Real Bunny HQ is {0} blocks away".format(hq_distance))

if __name__ == '__main__':
    data = util.get_input(__DAY__)
    solve(data)
