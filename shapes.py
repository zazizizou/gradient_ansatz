from numpy import sqrt, nan, arange
from numbers import Number
from scipy.optimize import curve_fit
from utils import gauss


class Circle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius


class Rectangle:
    def __init__(self, xmin=0, ymin=0, xmax=0, ymax=0):
        self.xmax = xmax
        self.ymax = ymax
        self.xmin = xmin
        self.ymin = ymin

        self.height = self.ymax - self.ymin
        self.width = self.xmax - self.xmin

    def by_width_height(self, xmin, ymin, width, height):
        self.xmin = xmin
        self.ymin = ymin
        self.height = height
        self.width = width

        self.xmax = self.xmin + self.width
        self.ymax = self.ymin + self.height

        return self

    def by_center_width_height(self, xcent, ycent, width, height):
        self.xmin = xcent - width / 2
        self.ymin = ycent - height / 2
        self.xmax = xcent + width / 2
        self.ymax = ycent + height / 2

        self.height = height
        self.width = width

    def __eq__(self, other):
        if (self.xmax == other.xmax and
            self.ymax == other.ymax and
            self.height == other.height and
            self.width == other.width ):
                return True
        else:
            return False

    def __ne__(self, other):
        if(self.xmax != other.xmax and
        self.ymax != other.ymax and
        self.height != other.height and
        self.width != other.width):
            return True
        else:
            return False


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __mul__(self, other):
        if isinstance(other, Number):
            return Point(other * self.x, other * self.y)

        else:
            return Point(other.x * self.x, other.y * self.y)

    def __rmul__(self, other):
        if isinstance(other, Number):
            return Point(other * self.x, other * self.y)
        else:
            return Point(other.x * self.x, other.y * self.y)

    def __add__(self, other):
        if isinstance(other, Number):
            return Point(other + self.x, other + self.y)
        else:
            return Point(other.x + self.x, other.y + self.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return self.x != other.x and self.y != other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def get_coord(self, dtype=None):
        if dtype == "int":
            return int(self.x), int(self.y)
        else:
            return self.x, self.y

    def has_negative_coord(self):
        return self.x < 0 or self.y < 0


def circle_to_rectangle(circ):
    return Rectangle(circ.x - circ.radius * sqrt(2),  # xmin
                     circ.y - circ.radius * sqrt(2),  # ymin
                     circ.x + circ.radius * sqrt(2),  # xmax
                     circ.y + circ.radius * sqrt(2))  # ymax


class Line:
    def __init__(self, direction, point, n_samples=10):
        self.direction = direction
        self.point = point


class CalibBubble:
    def __init__(self, line, radius, frame_start, frame_end):
        self.line = line
        self.radius = radius
        self.frame_start = int(frame_start)
        self.frame_end =   int(frame_end)
