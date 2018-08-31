#!/usr/bin/env python

import json
from collections import defaultdict
from copy import copy
from timer import Timer
from itertools import count, permutations, groupby
import argparse, hashlib, re
from string import ascii_lowercase

def day01(data):
  f = lambda l: l.count('(') - l.count(')')
  print("Part 1: Final floor = {0}".format(f(data)))
  print("Part 2: Reached -1 at {0}".format(next(i for i in count(1) if f(data[:i]) == -1)))

def day02(data):
  def paper(l,w,h):
    areas = [l*w, l*h, w*h]
    return 2*sum(areas) + min(areas)
  def ribbon(l,w,h):
    perimeters = [l+w, l+h, w+h]
    return l*w*h + 2*min(perimeters)
  solve_for = lambda f: sum([f(*map(int, l.split('x'))) for l in data.splitlines()])
  print("Part 1: {0} sqft of paper".format(solve_for(paper)))
  print("Part 2: {0} ft of ribbon".format(solve_for(ribbon)))

def day03(data):
  def solve(moves, house=(0,0)):
    move = lambda d, x, y: {'^':(x,y+1),'v':(x,y-1),'<':(x-1,y),'>':(x+1,y)}[d]
    houses = set([house])
    for direction in moves:
      house = move(direction, *house)
      houses.add(house)
    return houses
  print("Part 1: {0} different houses".format(len(solve(data))))
  print("Part 2: {0} different houses".format(len(solve(data[::2]) | solve(data[1::2]))))

def day04(data):
  def solve(key, n):
    md5 = lambda s: hashlib.md5(s.encode()).hexdigest()
    return next(i for i in count() if md5(key + str(i)).startswith('0'*n))
  print("Part 1: {0} causes 5 leading zeroes".format(solve(data, 5)))
  print("Part 2: {0} causes 6 leading zeroes".format(solve(data, 6)))

def day05(data):
  nice = lambda w: not re.search('(ab|cd|pq|xy)', w) and len(re.findall('[aeiou]', w)) > 2 and re.search(r'(.)\1', w)
  really_nice = lambda w: re.search(r'(..).*\1', w) and re.search(r'(.).\1', w)
  def solve(data, f):
    return len([line for line in data.splitlines() if f(line)])
  print("Part 1: {0} are nice".format(solve(data, nice)))
  print("Part 2: {0} are really nice".format(solve(data, really_nice)))

def day06(data):
  p = re.compile(".*(on|off|toggle) (\d+),(\d+).*?(\d+),(\d+)")
  def solve(data, f):
    grid = [[0 for n in range(1000)] for y in range(1000)]
    for op, x1, y1, x2, y2 in [(op, int(x1), int(y1), int(x2), int(y2)) for (op, x1, y1, x2, y2) in p.findall(data)]:
      for y in range(y1, y2+1):
        for x in range(x1, x2+1):
          grid[y][x] = f(op, grid[y][x])
    return sum([sum(col) for col in grid])
  f1 = lambda op, n: {'toggle': 1 if n == 0 else 0, 'on': 1, 'off': 0}[op]
  f2 = lambda op, n: {'toggle': n+2, 'on': n+1, 'off': max(0, n-1)}[op]
  print("Part 1: {0} lights are on".format(solve(data, f1)))
  print("Part 2: {0} total brightness".format(solve(data, f2)))

def day07(data):
  unary_op_re = re.compile('(NOT )?([a-z]+|[0-9]+) -> ([a-z]+|[0-9]+)')
  binary_op_re = re.compile('([a-z]+|[0-9]+) (AND|OR|LSHIFT|RSHIFT) ([a-z]+|[0-9]+) -> ([a-z]+|[0-9]+)')
  def solve(ops):
    while not type(ops['a']) == type(0):
      for k, expr in ops.items():
        try:
          ops[k] = eval(expr)
        except (ValueError, TypeError) as e:
          continue
    return ops['a']
  def coerce_int(x):
    try:
      x = int(x, 10)
    except ValueError:
      x = 'ops["{0}"]'.format(x)
    return x
  ops_dict = dict()
  for line in data.splitlines():
    if unary_op_re.match(line):
      op, a, b = unary_op_re.match(line).groups()
      a = coerce_int(a)
      ops_dict[b] = '~ {0}'.format(a) if op else '{0}'.format(a)
    elif binary_op_re.match(line):
      a, op, b, c = binary_op_re.match(line).groups()
      a = coerce_int(a)
      b = coerce_int(b)
      op = {'AND': ' & ', 'OR': ' | ', 'LSHIFT': ' << ', 'RSHIFT': ' >> '}[op]
      ops_dict[c] = '{0} {1} {2}'.format(a, op, b)  
  a = solve(copy(ops_dict))
  print("Part 1: a = {0} ".format(a))
  ops_dict['b'] = a
  print("Part 2: a = {0} when b = {1}".format(solve(ops_dict), a))

def day08(data):
  unnecessary = lambda line: len(line) - len(eval(line))
  extra = lambda line: 2 + line.count('\\') + line.count('"')
  print("Part 1: {0} unnecessary chars".format(sum(map(unnecessary, data.splitlines()))))
  print("Part 2: {0} extra chars".format(sum(map(extra, data.splitlines()))))

def day09(data):
  distances = defaultdict(dict)
  for source, dest, distance in re.findall('(\w+) to (\w+) = (\d+)', data):
    distances[source][dest] = int(distance)
    distances[dest][source] = int(distance)

  shortest_path, longest_path = None, None
  shortest_distance, longest_distance = None, None
  for path in permutations(distances.keys()):
    curr = sum(distances[source][dest] for source, dest in zip(path, path[1:]))
    if not shortest_distance or curr < shortest_distance:
      shortest_path, shortest_distance = path, curr
    if not longest_distance or curr > longest_distance:
      longest_path, longest_distance = path, curr

  print("Part 1: Shortest distance {0}".format(shortest_distance))
  print("Part 2: Longest distance {0}".format(longest_distance))

def day10(data):
  apply = lambda s: ''.join(['{0}{1}'.format(len(list(g)), c) for c, g in groupby(s)])
  for n in range(40): data = apply(data)
  print("Part 1: {0} digits".format(len(data)))
  for n in range(10): data = apply(data)
  print("Part 2: {0} digits".format(len(data)))

def day11(data):
  def valid(s):
    if len(set('iol') & set(s)) > 0: return False
    if not any([ascii_lowercase[n:n+3] in s for n in range(24)]): return False
    if sum([2 if len(list(y)) >= 4 else 1 for x,y in groupby(s) if len(list(y)) >= 2]) < 2: return False
    return True
  def inc(s):
    if s == '': return 'a'
    elif s[-1] < 'z': return s[0:-1] + chr(ord(s[-1])+1)
    else: return inc(s[:-1]) + 'a'
  while not valid(data): data = inc(data)
  print("Part 1: Next password is {0}".format(data))
  data = inc(data)
  while not valid(data): data = inc(data)
  print("Part 2: Next password is {0}".format(data))

def day12(data):
  def getnumbers(json):
      if type(json) == type(0): 
          return json
      if type(json) == type([]): 
          return sum(map(getnumbers, json))
      if type(json) == type({}) and not 'red' in json.values():
          return sum(map(getnumbers, json.values()))
      return 0
  print("Part 1: total: {0}".format(sum(map(int, re.findall('-?\d+', data)))))
  print("Part 2: non-red total: {0}".format(getnumbers(json.loads(data))))

def day13(data):
  def solve(rules, people):
    most_happiness = 0
    for perm in permutations(people):
      perm = list(perm)
      happiness = sum([rules[min(p1, p2)][max(p1, p2)] for p1, p2 in zip(perm, perm[1:]+[perm[0]])])
      if happiness > most_happiness:
        most_happiness = happiness
    return most_happiness
  happiness_dict = defaultdict(lambda: defaultdict(int))
  people = set()
  for p1, op, n, p2 in re.findall("(?P<p1>\w+).*?(?P<op>lose|gain) (?P<n>\d+).*?(?P<p2>\w+)\.", data):
    people.add(p1)
    happiness_dict[min(p1, p2)][max(p1, p2)] += int('-'+n if op == 'lose' else n)
  print("Part 1: {0} is most happiness".format(solve(happiness_dict, people)))
  for p1 in people:
    happiness_dict[p1]['Ratmatix'] = 0
  people.add('Ratmatix')
  print("Part 2: {0} is most happiness".format(solve(happiness_dict, people)))

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

  solutions = sorted([(day, func) for day, func in globals().items() if re.match('day\d+', day)])
  slow_solutions = set(['day04', 'day06'])

  if args.day:
    solutions = [(d, f) for d, f in solutions if d == "day{0:02}".format(args.day)]

  if not args.all:
    solutions = [solutions[-1]]

  for day, func in solutions:
    if args.skipslow and day in slow_solutions: continue
    print("Solution for {0}".format(day))
    with open('./inputs/{0}'.format(day)) as handle:
      data = handle.read().strip()
    with Timer() as timer:
      func(data)
