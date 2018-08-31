#!/usr/bin/env python

import re
from collections import defaultdict
import logging, sys

class Rectangle(object):
  def __init__(self, x1, y1, x2, y2):
    self.x1 = x1
    self.y1 = y1
    self.x2 = x2
    self.y2 = y2

    self.area = (x2-x1)*(y2-y1)
    self.state = 'off'
    self.brightness = 0

  def __repr__(self):
    return str("({0},{1}) -> ({2},{3}): currently {4}, brightness: {5}".format(self.x1, self.y1, self.x2, self.y2, self.state, self.brightness))

  def __contains__(self, x1, y1, x2, y2):
    return (x1 <= self.x1 < x2) and (y1 <= self.y1 <= y2)
    #return (self.x1 <= x < self.x2) and (self.y1 <= y < self.y2)

  def apply(self, op):
    if op == 'toggle':
      self.state = 'off' if self.state == 'on' else 'on'
      self.brightness += 2
    elif op == 'on':
      self.state = 'on'
      self.brightness += 1
    elif op == 'off':
      self.state = 'off'
      self.brightness = max(0, self.brightness-1)

  def stats(self):
    return self.area, self.state, self.brightness

''' For the purpose of coordinates gathering, top/left are inclusive, bottom/right are exclusive'''
def day06(data, x_max=1000, y_max=1000):
  p = re.compile(".*(on|off|toggle) (\d+),(\d+).*?(\d+),(\d+)")
  ops = [(op, (int(x1), int(y1), int(x2)+1, int(y2)+1)) for (op, x1, y1, x2, y2) in p.findall(data)]
  logger.warn("Found {0} operations".format(len(ops)))
  
  # Build set of all corners
  #all_corners = set([(0,0), (0,y_max), (x_max,0), (x_max,y_max)])
  xs_by_y = defaultdict(list)
  xs_by_y[0].extend([0, x_max])
  xs_by_y[y_max].extend([0, x_max])
  print(xs_by_y)
  for op, (x1, y1, x2, y2) in ops:
    #all_corners.update([(x1,y1), (x1,y2), (x2, y1), (x2, y2)])
    xs_by_y[y1].extend([x1, x2])
    xs_by_y[y2].extend([x1, x2])
  #for y in range(10):
  #  logger.debug(xs_by_y[y])

  rectangles = []
  # Now build a set of all rectangles by scanning horizontally
  for top_y in range(0, y_max):
    logger.info("Beginning to scan across row {0}".format(top_y))
    corners_in_y = list(filter(lambda x: x<x_max, sorted(xs_by_y[top_y], reverse=True))) 
    #logger.debug("Found {0} corners beginning in this row: {1}".format(len(corners_in_y), corners_in_y))
    while corners_in_y:
      left_x = corners_in_y.pop() # Pop because we'll never need this top left again, scanning across and down
      #logger.debug("Starting new rectangle, top left: ({0}, {1})".format(left_x, top_y))
      try:
        right_x = corners_in_y[-1] # Grab the coordinates of next corner in this row
      except IndexError: # Lucky! Can go all the way to end of row
        right_x = x_max
      #logger.info("top right x: {0}".format(right_x))
      
      for y2 in range(top_y+1, y_max):
        #logger.info("Scanning y={0} for bottom right conflict".format(y2))
        try:
          conflicting_corner = sorted([x for x in xs_by_y[y2] if x < right_x])[0]
        except IndexError: # Lucky, no conflincting corners in this row! 
          #logger.info("No conflicting rectangle! Continuing to next")
          continue # Next y2!
        #logger.info("Found a conflicting corner, at {0},{1}".format(conflicting_corner, y2))
        bottom_y = y2
        break
      logger.info("Adding a rectangle ({0},{1})->({2},{3})".format(left_x, top_y, right_x, bottom_y))
      #rectangles.append(((left_x, top_y), (right_x, bottom_y)))
      rectangles.append(Rectangle(left_x, top_y, right_x, bottom_y))
      logger.debug("Adding artificial corner below bottom left")
      xs_by_y[bottom_y].append(left_x)

  print("Created {0} rectangles".format(len(rectangles)))
  #print("Rectangles: {0}".format(rectangles))
  print("Now iterate over the operations. Check for top left corner membership in rectangles..?")
  for op, (x1, y1, x2, y2) in ops:
    intersected = 0
    logger.info("Applying {0} to ({1},{2})->({3},{4})".format(op, x1, y1, x2, y2))
    for rectangle in rectangles:
      if (x1, y1, x2, y2) in rectangle: # Yes this is backwards, but it needs to be ...
        intersected += 1
      
      #if (x1, y1) in rectangle:
        #logger.info("Rectangle intersects: {0}".format(rectangle))
        rectangle.apply(op)
    print("{0} rectangles intersected with the op".format(intersected))

  brightness = 0
  on = 0
  for rectangle in rectangles:
    area, rec_state, rec_brightness = rectangle.stats()
    brightness += (area * rec_brightness)
    on += (area if rec_state == 'on' else 0)
  print("{0} are on, total brightness is {1}".format(on, brightness))
    
        
def init_logging(level):
  level = [logging.FATAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][level]
  global logger
  logger = logging.getLogger('adventofcodeday06')
  logger.setLevel(level)
  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.DEBUG)
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  logger.addHandler(ch)
    
if __name__ ==  '__main__':
  import argparse
  parser = argparse.ArgumentParser("Advent of Code")
  parser.add_argument("-v", "--verbose", action="count", default=0)
  args = parser.parse_args()
  init_logging(args.verbose)
  with open('../inputs/day06') as f:
    data = f.read()
  day06(data)
