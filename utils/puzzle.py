import os.path

from timer import Timer


class Puzzle(object):
    """
    Container class for all the solutions - implement solveA and solveB for the solution objects that extend this object
    """
    slow = False

    def __init__(self):
        print("Solutions for day{:02}".format(self.day))
        if type(self) == Puzzle:
            raise Exception("<Puzzle> must be subclassed.")
        self.data = self.get_data()

    @property
    def day(self):
        return int(self.__class__.__name__.replace('Puzzle', ''))

    def get_data(self):
        input_path = 'inputs/day{:02}'.format(self.day)
        if not os.path.exists(input_path):
            raise FileNotFoundError('Input for day {} does not exist at: {}'.format(self.day, input_path))
        with open(input_path) as f:
            data = f.read()
        return data

    def solve(self):
        with Timer():
            print("A: {}".format(self.solve_a()))
        with Timer():
            print("B: {}".format(self.solve_b()))

    def solve_a(self):
        raise NotImplementedError(".solveA() for day {:02} has not been implemented yet".format(self.day))

    def solve_b(self):
        raise NotImplementedError(".solveB() for day {:02} has not been implemented yet".format(self.day))