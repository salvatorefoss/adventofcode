#!/bin/env python

import argparse
from collections import Counter, defaultdict, deque, namedtuple, OrderedDict
import inspect
from functools import reduce
from itertools import combinations, count, islice, takewhile
import math
from operator import itemgetter
import operator
import re
from string import ascii_lowercase
import sys

from utils.puzzle import Puzzle
from utils.timer import Timer


class Puzzle01(Puzzle):
    """ Day 1: Inverse Captcha """
    @staticmethod
    def algorithm(pairs):
        """
        Not much to it, sum the numbers that have the matching pair
        """
        return sum(int(a) for (a, b) in pairs if a == b)

    def solve_a(self):
        """
        Pair each number with the number after it (cycled around)
        """
        return self.algorithm(zip(self.data, self.data[1:] + self.data[:1]))

    def solve_b(self):
        """
        Pair each number with the number half way around the list - only need to check first half, then double
        """
        return 2 * self.algorithm(zip(self.data[len(self.data) // 2:], self.data[:len(self.data) // 2]))


class Puzzle02(Puzzle):
    """ Day 2: Corruption Checksum """
    def solve_a(self):
        """ Sum the difference between the max and min of each line """
        return sum(max(values) - min(values) for values in ([int(value) for value in line.split()] for line in self.data.splitlines()))

    def solve_b(self):
        """ Generate all pairings of the nubmers on each line, find the evenly divisible pair, then use their division for the sum """
        return sum(max(a, b) // min(a, b) for line in self.data.splitlines() for (a, b) in combinations(map(int, line.split()), 2) if not max(a, b) % min(a, b))


class A141481(object):
    """
    So this sequence is actually known to the Online Encyclopedia of Integer Sequences as:
    "A141481 - Square spiral of sums of selected preceding terms, starting at 1"

    Algorithm is - travel in the direction of travel until the cell in the next direction is null,
    then change to that direction, setting the value of the current cell to the sum of its neighbours

    Using a two dimension defaultdict means you don't need to check for existance. the sequence is monotonic so 0 must
    means the cell is uninitialised.
    """
    point = namedtuple('point', ['x', 'y'])
    directions = deque([point(0, -1), point(1, 0), point(0, 1), point(-1, 0)])
    ds = (-1, 0, 1)

    def __init__(self):
        self.grid = defaultdict(lambda: defaultdict(int))
        self.x = 0
        self.y = 0
        self.grid[self.x][self.y] = 1

    def update_direction(self):
        if not self.grid[self.x + self.directions[1].x][self.y + self.directions[1].y]:
            self.directions.rotate(-1)

    def move(self):
        self.x, self.y = self.x + self.directions[0].x, self.y + self.directions[0].y

    def __iter__(self):
        yield self.current_score
        while True:
            self.update_direction()
            self.move()
            self.calculatore_score_for_current_coords()
            yield self.current_score

    @property
    def current_score(self):
        return self.grid[self.x][self.y]

    def calculatore_score_for_current_coords(self):
        self.grid[self.x][self.y] = sum(self.grid[self.x + x][self.y + y] for x in self.ds for y in self.ds if x or y)


class Puzzle03(Puzzle):
    """ Day 3: Spiral Memory """
    def solve_a(self):
        """
        Calculate the size of the square:
        The number of steps is the number of layers + how many steps to the centre of the side
        """
        n = int(self.data)
        width = int(math.ceil(math.sqrt(n)))
        if not width % 2:
            width += 1
        layers = width // 2
        corners = list(range(width ** 2, width ** 2 - 4 * (width - 1), -(width - 1)))
        for corner in corners[1:]:
            if (corner - n + 1) < width:
                distance_to_centre_of_side = abs(n - (corner - width // 2))
                return layers + distance_to_centre_of_side

    def solve_b(self):
        """
        Don't know how to do this without generating the numbers. Use the generator class I wrote above
        """
        return next(v for v in A141481() if v > int(self.data))


class Puzzle04(Puzzle):
    """ Day 4: High-entropy Passphrases"""
    @staticmethod
    def algorithm(pairs):
        """
        Return the number of lines in which the words and the computed words are equal (none are 'duplicates')
        """
        return sum(1 for (words, checked_words) in pairs if len(words) == len(set(checked_words)))

    def solve_a(self):
        """
        Pretty simple, just need the words themselves
        """
        return self.algorithm((line.split(), line.split()) for line in self.data.splitlines())

    def solve_b(self):
        """
        Sort the letters in each of the words
        """
        def get_sorted_words(line):
            return [''.join(sorted(word)) for word in line.split()]
        return self.algorithm((line.split(), get_sorted_words(line)) for line in self.data.splitlines())


class Puzzle05(Puzzle):
    """ Day 5: A Maze of Twisty Trampolines, All Alike """
    def algorithm(self, func):
        """
        Keep all the offsets in a list, jump around the list until you try to jump outside of the bounds of the list
        Perform the func() on the offset you just jumped from to change its offset value.
        """
        jump_table = [int(line) for line in self.data.splitlines()]
        index = 0
        for n in count():
            try:
                jump_table[index], index = func(jump_table[index]), index + jump_table[index]
            except IndexError:
                return n

    def solve_a(self):
        """ Calculate the jumps where the jump is increased by one after performing it """
        return self.algorithm(lambda offset: offset + 1)

    def solve_b(self):
        """ Calculate the jumps where the jump is increased by 1 if it was <3, otherwise decreased by 1 """
        return self.algorithm(lambda offset: offset + 1 if offset < 3 else offset - 1)


class MemoryAllocation(object):
    """
    A list of all the memory banks and their block counts
    Maintains a dictionary of the state of the memory banks and when we've seen that state
    """
    def __init__(self, data):
        self.memory_banks = [int(blocks) for blocks in data.split()]
        self.seen_states = {tuple(self.memory_banks): 0}
        self.redistributions = 0

    def get_largest_memory_bank(self):
        return sorted(enumerate(self.memory_banks), key=itemgetter(1), reverse=True)[0][0]

    def redistribute(self, index):
        """
        Redistribute the given memory bank by dividing its number of blocks by the number of memory banks
        The quotient determines what to add to all registers and the remainder determines the number of memory banks
        after the given index to increase by 1.
        """
        blocks = self.memory_banks[index]
        self.memory_banks[index] = 0
        block_div, block_mod = divmod(blocks, len(self.memory_banks))
        if block_div:
            self.memory_banks = [memory_bank + block_div for memory_bank in self.memory_banks]
        if block_mod:
            for memory_bank_index in range(index + 1, index + block_mod + 1):
                self.memory_banks[memory_bank_index % len(self.memory_banks)] += 1
        self.redistributions += 1

    def in_infinite_loop(self):
        state = tuple(self.memory_banks)
        if state in self.seen_states:
            return True
        self.seen_states[state] = self.redistributions
        return False

    def trigger_infinite_loop(self):
        while True:
            self.redistribute(self.get_largest_memory_bank())
            if self.in_infinite_loop():
                return self.redistributions, self.redistributions - self.seen_states[tuple(self.memory_banks)]


class Puzzle06(Puzzle):
    """ Day 6: Memory Reallocation """
    def __init__(self):
        super().__init__()
        with Timer():
            self.memory_allocation = MemoryAllocation(self.data)

    def solve_a(self):
        """ Create the Memory Allocation Table, trigger the infinite loop and return the number of redistributions """
        return self.memory_allocation.trigger_infinite_loop()[0]

    def solve_b(self):
        """ Create the Memory Allocation Table, trigger the infinite loop and return when you last saw the state we're in """
        return self.memory_allocation.trigger_infinite_loop()[1]


class Program(object):
    def __init__(self, name, weight, below):
        self.name = name
        self.weight = int(weight)
        self.supporting = set(below.split(', ')) if below else set()


class ProgramTower(object):
    """
    Builds a tree of the Programs where the children are the programs in the supporting set
    """
    def __init__(self, programs, name=None):
        self.program = programs[name if name else self.find_root(programs)]
        self.children = [ProgramTower(programs, name=child) for child in self.program.supporting]

    @property
    def total_weight(self):
        return self.program.weight + sum(c.total_weight for c in self.children)

    def get_unbalanced_child(self, unbalanced_weight):
        return next(child for child in self.children if child.total_weight == unbalanced_weight)

    def find_wrong_weight(self, delta=None):
        """
        Finding the incorrect weight counts the number of unique subtower weights, recursing through the odd
        one out until they are balanced, when it returns its weight minus the difference between the normal and abnormal
        children
        """
        child_weights = Counter(c.total_weight for c in self.children)
        if len(child_weights) > 1:
            unbalanced_weight = next(reversed(child_weights.most_common()))[0]
            return self.get_unbalanced_child(unbalanced_weight).find_wrong_weight(max(child_weights) - min(child_weights))
        return self.program.weight - delta

    @staticmethod
    def find_root(programs):
        return reduce(set.difference, (program.supporting for program in programs.values()), set(programs)).pop()


class Puzzle07(Puzzle):
    """ Day 7: Recursive Circus """
    def __init__(self):
        super().__init__()
        with Timer():
            """ Create a dictionary of programs keyed by their name """
            matcher = re.compile(r'^([a-z]+)\s\((\d+)\)(?: -> ((?:(?:\w+)(?:, )?)*))?$', re.M)
            programs = {program.name: program for program in (Program(*match.groups()) for match in matcher.finditer(self.data))}
            self.program_tower = ProgramTower(programs)

    def solve_a(self):
        """ Create the ProgramTower and return the name of the root """
        return self.program_tower.program.name

    def solve_b(self):
        """
        Create the Program Tower and recurse down through the unbalanced towers then return the difference
        of its weight and the difference between the unbalanced towers
        """
        return self.program_tower.find_wrong_weight()


class Instruction(object):
    """
    An Instruction is built of three things - the register it affects, the delta by which it affects the register,
    and a function which determines in the register should be affected. The function takes a single argument; all the
    registers, and uses the already known index.
    Each time you apply an instruction, the resulting value is returned, which is being used to determine the largest
    value seen
    """
    def __init__(self, register, delta, value, cmp_r, cmp, cmp_v):
        self.register = register
        self.delta = (1 if delta == 'inc' else -1) * int(value)

        self.condition = lambda registers: {
            '<': operator.lt,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne,
            '>': operator.gt,
            '>=': operator.ge
        }[cmp](registers[cmp_r], int(cmp_v))

    def apply(self, registers):
        if self.condition(registers):
            registers[self.register] += self.delta
        return registers[self.register]


class Puzzle08(Puzzle):
    """ Day 8: I Heard You Like Registers """
    def __init__(self):
        super().__init__()
        with Timer():
            """
            The registers are defined as a default dictionary which initialises them to 0
            The instructions are then applied to the registers
            """
            matcher = re.compile(r'(\w+) (inc|dec) (-?\d+) if (\w+) (.*?) (-?\d+)')
            instructions = [Instruction(*match.groups()) for match in matcher.finditer(self.data)]
            self.registers = defaultdict(int)
            self.max_value = max(instruction.apply(self.registers) for instruction in instructions)

    def solve_a(self):
        """ Return the largest value in any of the registers after all the instructions have been applied """
        return Counter(self.registers).most_common(1)[0][1]

    def solve_b(self):
        """ The max value is kept track of by returning the register value after setting it """
        return self.max_value


class Stream(object):
    """
    A stream is composed of Ignored, Garbage and Group items.
    Pop the first item off the stream, determine how it should be handled, and create a recursive tree of
    the items.
    Calculate the score and size by determining which subclass the scoring and size methods are being run in.
    """
    def __init__(self):
        self.contents = []
        self.ignored = []
        self.score = 0

    def process(self, data):
        assert data.popleft() == '{'
        self.contents.append(Group(data))

    @property
    def total_score(self):
        return self.score + sum(content.total_score for content in self.contents if isinstance(content, Group))

    @property
    def size(self):
        if isinstance(self, Garbage):
            return len(self.contents) + sum(content.size for content in self.contents if isinstance(content, (Group, Garbage)))
        else:
            return sum(content.size for content in self.contents if isinstance(content, (Group, Garbage)))


class Ignored(Stream):
    def __init__(self, data):
        super().__init__()
        self.contents.append(data.popleft())


class Garbage(Stream):
    def __init__(self, data):
        super().__init__()
        while True:
            value = data.popleft()
            if value == '!':
                self.ignored.append(Ignored(data))
            elif value == '>':
                break
            else:
                self.contents.append(value)


class Group(Stream):
    def __init__(self, data, score=1):
        super().__init__()
        self.score = score

        while True:
            value = data.popleft()
            if value == '{':
                self.contents.append(Group(data, self.score + 1))
            elif value == '<':
                self.contents.append(Garbage(data))
            elif value == '}':
                break
            else:
                self.contents.append(value)


class Puzzle09(Puzzle):
    """ Day 9: Stream Processing """
    def __init__(self):
        super().__init__()
        with Timer():
            self.stream = Stream()
            self.stream.process(deque(self.data))

    def solve_a(self):
        return self.stream.total_score

    def solve_b(self):
        return self.stream.size


class KnotHash(list):
    suffix = [17, 31, 73, 47, 23]

    def __init__(self, lengths=None, size=256, rounds=64, use_suffix=True):
        super().__init__(range(size))
        self.rounds = rounds
        self.use_suffix = use_suffix
        self.index = 0
        self.skip_size = 0
        if lengths:
            self.apply(lengths)

    def apply(self, lengths):
        if self.use_suffix:
            lengths += self.suffix

        for _ in range(self.rounds):
            for length in lengths:
                self.reverse_sublist(length)
                self.index = self.index + length + self.skip_size
                self.skip_size += 1

    def reverse_sublist(self, length):
        """
        Reverse each specified sublist - wrap around is taken care of by overriding methods below to mod the index
        """
        for n, i in enumerate(range(self.index, self.index + length // 2), start=1):
            self[i], self[self.index + length - n] = self[self.index + length - n], self[i]

    def dense_hash(self, group_size=16):
        """ Reduce each group of X bytes by xoring them together """
        return [reduce(operator.xor, group) for group in zip(*[iter(self)] * group_size)]

    def __str__(self):
        return ''.join(f'{c:02x}' for c in self.dense_hash())

    def __getitem__(self, index):
        return super().__getitem__(index % len(self))

    def __setitem__(self, index, value):
        return super().__setitem__(index % len(self), value)


class Puzzle10(Puzzle):
    """ Day 10: Knot Hash """
    def solve_a(self):
        knot = KnotHash(list(map(int, self.data.split(','))), rounds=1, use_suffix=False)
        return knot[0] * knot[1]

    def solve_b(self):
        knot = KnotHash(list(map(ord, self.data)))
        return str(knot)


class HexGrid(object):
    """ Hexagonal Coordinate System can easily be represented as a cube coordinate system """
    direction_mapping = {
        'n':  (0, 1, -1), 's':  (0,  -1, 1),
        'ne': (1, 0, -1), 'sw': (-1, 0,  1),
        'nw': (-1, 1, 0), 'se': (1, -1,  0)
    }

    def __init__(self):
        self.x, self.y, self.z = 0, 0, 0
        self.furthest = 0

    def move(self, direction):
        """ Each time you move, calculate if this is the furthest we've been from the origin """
        dx, dy, dz = self.direction_mapping[direction]
        self.x, self.y, self.z = self.x + dx, self.y + dy, self.z + dz
        self.furthest = max(self.furthest, self.distance_from_origin())

    def distance_from_origin(self):
        """ In a cube coordinate system, the distance between two points are the difference between coords div by 2 """
        return (abs(self.x) + abs(self.y) + abs(self.z)) // 2


class Puzzle11(Puzzle):
    """ Day 11: Hex Ed """
    def __init__(self):
        super().__init__()
        with Timer():
            self.hex_grid = HexGrid()
            for direction in self.data.split(','):
                self.hex_grid.move(direction)

    def solve_a(self):
        return self.hex_grid.distance_from_origin()

    def solve_b(self):
        return self.hex_grid.furthest


class Village(object):
    """
    Make Groups connected to a starting node, removing connected nodes from the available pool
    Repeat until all nodes are exhausted
    """
    def __init__(self, programs):
        self.programs = programs
        self.groups = []

    def grouped(self, to):
        group = {to}
        self.groups.append(group)
        unexplored = set(self.programs.pop(to))
        while len(unexplored):
            curr = unexplored.pop()
            group.add(curr)
            unexplored.update(self.programs.pop(curr, []))
        return group

    def all_groups(self):
        while self.programs:
            self.grouped(next(iter(self.programs.keys())))
        return self.groups


class Puzzle12(Puzzle):
    """ Day 12: Digital Plumber """
    def __init__(self):
        super().__init__()
        with Timer():
            matcher = re.compile(r'^(\d+)(?: <-> ((?:(?:\d+)(?:, )?)*))?$', re.M)
            programs = {pid: pids.split(', ') for pid, pids in (match.groups() for match in matcher.finditer(self.data))}
            self.village = Village(programs)

    def solve_a(self):
        return len(self.village.grouped('0'))

    def solve_b(self):
        return len(self.village.all_groups())


class Layer(object):
    def __init__(self, depth, scanner_range, current_position=0, delta=1):
        self.depth = depth
        self.scanner_range = scanner_range
        self.current_position = current_position
        self.delta = delta

    def proceed(self):
        if self.delta == 1 and self.current_position == self.scanner_range - 1:
            self.delta = -1
        elif self.delta == -1 and self.current_position == 0:
            self.delta = 1
        self.current_position += self.delta

    def clone(self):
        return Layer(self.depth, self.scanner_range, self.current_position, self.delta)


class Firewall(object):
    def __init__(self, layers):
        self.layers = layers

    def proceed(self):
        for layer in self.layers:
            layer.proceed()

    def traverse(self, fail_early=False):
        severity = 0
        for current_position in count():
            for layer in self.layers:
                if current_position == layer.depth and layer.current_position == 0:
                    severity += layer.depth * layer.scanner_range
                    if fail_early:
                        return None
            if current_position == self.layers[-1].depth:
                return severity
            self.proceed()

    def clone(self):
        return Firewall([layer.clone() for layer in self.layers])


class Puzzle13(Puzzle):
    """
    Day 13: Packet Scanners

    I have a feeling there's a faster way to do this by finding a starting value (N) for when the scanner is not at 0
    for each layer in N picoseconds, instead of simulating the firewall after each picosecond...

    Will take a look at it next time I'm doing these!
    """
    slow = True

    def __init__(self):
        super().__init__()
        with Timer():
            matcher = re.compile(r'^(\d+): (\d+)$', re.M)
            self.firewall = Firewall([Layer(int(k), int(v)) for (k, v) in matcher.findall(self.data)])

    def solve_a(self):
        return self.firewall.clone().traverse()

    def solve_b(self):
        for initial_delay in count(start=1):
            self.firewall.proceed()
            if self.firewall.clone().traverse(fail_early=True) is not None:
                return initial_delay


class Bit(object):
    def __init__(self, row, col, disk, bit):
        self.row = row
        self.col = col
        self.disk = disk
        self.bit = bit == '1'
        self.label = 0

    def first_pass(self):
        if not self.bit:
            return
        a, b = 0, 0
        if not self.row == 0 and not self.col == 0:
            a, b = self.disk.bits[self.row - 1][self.col].label, self.disk.bits[self.row][self.col - 1].label
        elif self.row == 0:
            a = self.disk.bits[self.row][self.col-1].label
        elif self.col == 0:
            b = self.disk.bits[self.row-1][self.col].label

        if a and b:
            self.label = self.disk.associate(a, b)
        elif not a and not b:
            self.label = next(self.disk.label_counter)
        else:
            self.label = a if a else b

    def second_pass(self):
        while self.label in self.disk.associated_labels:
            self.label = min(self.disk.associated_labels[self.label])


class Disk(object):
    """
    So this uses the two-pass Connected-component labelling algorithm...
    Not worth commenting since it's on the wiki page, but I've stripped the code down to as little as I can.
    https://en.wikipedia.org/wiki/Connected-component_labeling
    """
    def __init__(self, key, size=128):
        self.bits = []
        self.label_counter = count(start=1)
        self.associated_labels = defaultdict(set)
        for row, kh in enumerate(KnotHash(list(map(ord, f'{key}-{n}'))) for n in range(size)):
            self.bits.append([Bit(row, col, self, bit) for col, bit in enumerate(bin(int(str(kh), 16))[2:].zfill(size))])

    def associations_of(self, label):
        s = {label}
        for a in self.associated_labels[label]:
            s.update(self.associations_of(a))
        return s

    def associate(self, a, b):
        cs = self.associations_of(a).union(self.associations_of(b))
        a = min(cs)
        self.associated_labels.pop(a, None)
        for l in filter(lambda x: not x == a, cs):
            self.associated_labels[l].add(a)
        return a

    @property
    def used(self):
        return sum(1 for bits in self.bits for bit in bits if bit.bit)

    @property
    def regions(self):
        return len({bit.label for bits in self.bits for bit in bits if not bit.label == 0})

    def groups(self):
        for row in self.bits:
            for bit in row:
                bit.first_pass()

        for row in self.bits:
            for bit in row:
                bit.second_pass()


class Puzzle14(Puzzle):
    """ Day 14: Disk Defragmentation """
    slow = True

    def __init__(self):
        super().__init__()
        with Timer():
            self.disk = Disk(self.data)

    def solve_a(self):
        return self.disk.used

    def solve_b(self):
        self.disk.groups()
        return self.disk.regions


class DuelingGenerator(object):
    def __init__(self, starting_value, factor, multiple_of=None):
        self.value = starting_value
        self.factor = factor
        self.multiple_of = multiple_of

    def __iter__(self):
        while True:
            self.value = (self.value * self.factor) % 2147483647
            while self.multiple_of and not self.value % self.multiple_of == 0:
                self.value = (self.value * self.factor) % 2147483647
            yield bin(self.value)[2:].zfill(16)[-16:]


class GeneratorA(DuelingGenerator):
    def __init__(self, starting_value, multiple_of=None):
        super().__init__(starting_value, 16807, multiple_of)


class GeneratorB(DuelingGenerator):
    def __init__(self, starting_value, multiple_of=None):
        super().__init__(starting_value, 48271, multiple_of)


class Puzzle15(Puzzle):
    """
    Day 15: Dueling Generators

    There has to be a faster way to do part a, will revisit in the future.
    """
    slow = True

    def solve_a(self):
        generator_a = GeneratorA(int(re.compile('A starts with (\d+)').search(self.data).group(1)))
        generator_b = GeneratorB(int(re.compile('B starts with (\d+)').search(self.data).group(1)))
        return sum(1 for x, y in islice(zip(generator_a, generator_b), 40000000) if x == y)

    def solve_b(self):
        generator_a = GeneratorA(int(re.compile('A starts with (\d+)').search(self.data).group(1)), 4)
        generator_b = GeneratorB(int(re.compile('B starts with (\d+)').search(self.data).group(1)), 8)
        return sum(1 for x, y in islice(zip(generator_a, generator_b), 5000000) if x == y)


class Dance(object):
    def __init__(self, dance_moves, num_dancers=16):
        self.dancers = deque(ascii_lowercase[:num_dancers])
        self.dance_moves = list(map(self.build_dance_move, dance_moves.split(',')))
        self.cycle = [tuple(self.dancers)]

    def apply(self):
        [dance_move() for dance_move in self.dance_moves]
        return self.dancers

    def build_cycle(self):
        self.cycle.append(tuple(self.dancers))
        while tuple(self.dancers) != self.cycle[0]:
            self.cycle.append(tuple(self.apply()))
        self.cycle.pop(-1)

    def build_dance_move(self, dance_move):
        return {
            's': lambda: self.spin(int(dance_move[1:])),
            'x': lambda: self.exchange(*map(int, dance_move[1:].split('/'))),
            'p': lambda: self.exchange(*map(self.dancers.index, dance_move[1:].split('/')))
        }[dance_move[0]]

    def spin(self, size):
        self.dancers.rotate(size % len(self.dancers))

    def exchange(self, a, b):
        self.dancers[a], self.dancers[b] = self.dancers[b], self.dancers[a]


class Puzzle16(Puzzle):
    """
    Day 16: Permutation Promenade

    Part 2 banks on the idea that eventually the dancers return to their original position after some amount of dancing
    When they do, you have a cycle length, take the modulo of a billion and the cycle length, which is effectively
    how many times the dancers dance to get into the resulting position.
    """
    def __init__(self):
        super().__init__()
        self.dance = Dance(self.data)

    def solve_a(self):
        return ''.join(self.dance.apply())

    def solve_b(self):
        self.dance.build_cycle()
        return ''.join(self.dance.cycle[1000000000 % len(self.dance.cycle)])


'''
class PuzzleN(Puzzle):
    """ Day N: Name """
    def __init__(self):
        super().__init__()
        with Timer():
            return
    
    @staticmethod # or not, depends on self.
    def algorithm(data):
        return

    def solve_a(self):
        return 

    def solve_b(self):
        return
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advent of Code solutions")
    parser.add_argument('-a', "--all", action="store_true")
    parser.add_argument('-s', "--skipslow", action="store_true")
    parser.add_argument('-d', "--day", action="store", type=int)
    args = parser.parse_args()

    puzzle_solutions = inspect.getmembers(sys.modules[__name__], lambda cls: inspect.isclass(cls) and issubclass(cls, Puzzle) and not cls == Puzzle)

    solutions = OrderedDict(sorted(puzzle_solutions))

    if args.day:
        solutions['Puzzle{:02}'.format(args.day)]().solve()
    elif args.all:
        for puzzle, solution in solutions.items():
            if args.skipslow and solution.slow:
                print("Skipping {} because it is a slow puzzle".format(puzzle))
                continue
            solution().solve()
    else:
        solutions[next(reversed(solutions))]().solve()
