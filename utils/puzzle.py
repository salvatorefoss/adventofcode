import os.path

from timer import Timer


class Puzzle(object):
    """
    Container class for all the solutions - implement solveA and solveB for the solution objects that extend this object
    """
    slow = False

    def __init__(self):
        if type(self) == Puzzle:
            raise Exception("<Puzzle> must be subclassed.")

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
        data = self.get_data()
        print("Solutions for day{:02}".format(self.day))
        with Timer():
            print("A: {}".format(self.solveA(data)))
        with Timer():
            print("B: {}".format(self.solveB(data)))

    def solveA(self, data):
        raise NotImplementedError(".solveA() for day {:02} has not been implemented yet".format(self.day))

    def solveB(self, data):
        raise NotImplementedError(".solveB() for day {:02} has not been implemented yet".format(self.day))