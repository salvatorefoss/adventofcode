#!/usr/bin/env python

from collections import defaultdict
from timer import Timer
from itertools import count
import argparse, hashlib, re

def day01(f):
  floor, basement = 0, 0
  for n, c in enumerate(f.readline(), 1):
    if c == '(': floor += 1
    elif c == ')': floor -= 1
    if floor == -1 and not basement:
      basement = n
  print("Part 1: Final floor = {0}".format(floor))
  print("Part 2: Reached -1 at {0}".format(basement))

def day02(f):
  paper, ribbon = 0, 0
  for (l, w, h) in (map(int, line.split('x')) for line in f):
    areas = [l*w, l*h, w*h]
    paper +=  2*sum(areas) + min(areas)
    perimeters = [l+w, l+h, w+h]
    ribbon += l*w*h + 2*min(perimeters)
  print("Part 1: {0} sqft of paper".format(paper))
  print("Part 2: {0} ft of ribbon".format(ribbon))

def day03(f):
  class santa(object):
    def __init__(self):
      self.x = 0
      self.y = 0
      self.houses = set([(0,0)])

    def move(self, d):
      if   d == '^': self.y += 1
      elif d == 'v': self.y -= 1
      elif d == '>': self.x += 1
      elif d == '<': self.x -= 1
      self.houses.add((self.x, self.y))

  santa, santa2, robosanta = santa(), santa(), santa()
  directions = iter(f.readline())
  for direction in directions:
    santa.move(direction)
    santa2.move(direction)
    direction = next(directions)
    santa.move(direction)
    robosanta.move(direction)

  print("Part 1: {0} different houses".format(len(santa.houses)))
  print("Part 2: {0} different houses".format(len(santa2.houses | robosanta.houses)))

def day05(f):
  p1 = re.compile(r'(.)\1')
  p2 = re.compile(r'([aeiou].*){3,}')
  p3 = re.compile('(ab|cd|pq|xy)')
  p4 = re.compile(r'(..).*\1')
  p5 = re.compile(r'(.).\1')
  nice = 0
  really_nice = 0
  for line in f:
    if p1.search(line) and p2.search(line) and not p3.search(line): 
      nice += 1
    if p4.search(line) and p5.search(line): 
      really_nice += 1
  print("Part 1: {0} are nice".format(nice))
  print("Part 2: {0} are really nice".format(really_nice))

def day08(f):
  unnecessary, extra = 0, 0
  for line in f:
    unnecessary += (len(line.strip()) - len(eval(line)))
    extra += 2 + line.count('\\') + line.count('"')
  print("Part 1: {0} unnecessary chars".format(unnecessary))
  print("Part 2: {0} extra chars".format(extra))

'''
def day00(data):
  def solve(data):
    pass
  print("Part 1: {0}".format(solve(data)))
  print("Part 2: {0}".format(solve(data)))
'''

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Advent of Code solutions")
  parser.add_argument('-a', "--all", action="store_true")
  parser.add_argument('-s', "--skipslow", action="store_true")
  parser.add_argument('-d', "--day", action="store", type=int)
  args = parser.parse_args()

  solutions = sorted([(d, f) for d, f in globals().items() if re.match('day\d+', d)])
  slow_solutions = set(['day04', 'day06'])

  if args.day:
    solutions = [(d, f) for d, f in solutions if d == "day{0:02}".format(args.day)]
    
  if not args.all:
    solutions = [solutions[-1]]

  for day, func in solutions:
    if args.skipslow and day in slow_solutions: continue
    print('"Faster" Solution for {0}'.format(day))
    with open('./inputs/{0}'.format(day)) as f:
      with Timer() as timer:
        func(f)
