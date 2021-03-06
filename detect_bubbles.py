import sys
sys.path.append('/Users/Habib/Google Drive/Uni Heidelberg/12 SS 2018/Masterarbeit/gradient_ansatz')

import numpy as np
from cv2 import imread, threshold, THRESH_BINARY
from skimage.feature import peak_local_max
from skimage import feature
from skimage.filters import scharr, sobel
from skimage.transform import hough_circle, hough_circle_peaks
from copy import deepcopy
from shapes import *
from scipy.optimize import curve_fit, leastsq
from scipy.misc import imresize
import pickle
import utils
from evaluate import inter_area
from scipy.signal import argrelextrema, find_peaks_cwt
from scipy.ndimage.filters import gaussian_filter1d
from classify_bubble import detec_bubble

"""
Coordinate system is:
.-–-> y
|
|
v
x
"""


def arr_to_vec(arr):
    return arr.reshape((arr.shape[0] * arr.shape[1],))


def resize_1d(signal, nb_points, interp="bilinear"):
    signal = np.asarray(signal)
    result = np.reshape(signal, (1, len(signal)))
    result = imresize(result, (1, nb_points), interp=interp)
    return result.reshape((nb_points))


def gauss(x, a, mu, sigma):
    return a * np.exp(- (x - mu)**2/sigma**2)


def get_neighbours(p, part="all"):
    neighbours = []
    if part == "upper" or part == "all":
        neighbours += [Point(p.x,   p.y+1),
                       Point(p.x-1,   p.y),
                       Point(p.x+1, p.y+1),
                       Point(p.x-1, p.y+1)]
    elif part == "lower" or part == "all":
        neighbours += [Point(p.x,   p.y+1),
                       # Point(p.x,   p.y-1),
                       Point(p.x+1, p.y),
                       # Point(p.x+1, p.y-1),
                       Point(p.x+1, p.y+1)]
    elif part == "lower_right":
        neighbours += [Point(p.x,     p.y + 1),
                       # Point(p.x - 1, p.y),
                       Point(p.x - 1, p.y + 1),
                       Point(p.x + 1, p.y + 1),
                       Point(p.x + 1, p.y)
                       ]
    elif part == "right":
        neighbours += [Point(p.x,     p.y + 1),
                       Point(p.x - 1, p.y),
                       Point(p.x - 1, p.y + 1),
                       Point(p.x + 1, p.y + 1),
                       Point(p.x + 1, p.y)]
    elif part == "upper_right":
        neighbours += [Point(p.x, p.y + 1),
                       Point(p.x - 1, p.y),
                       Point(p.x - 1, p.y + 1),
                       Point(p.x + 1, p.y + 1),
                       # Point(p.x + 1, p.y)
                        ]

    return neighbours


def get_upper_neighbours(p):
    return get_neighbours(p, part="upper")


def get_lower_neighbours(p):
    return get_neighbours(p, part="lower")


def get_lower_right_neighbours(p):
    return get_neighbours(p, part="lower_right")


def get_right_neighbours(p):
    return get_neighbours(p, part="right")


def get_neighbouring_pixels(p):
    px = set()
    px.add(Point(int(np.floor(p.x)), int(np.floor(p.y))))
    px.add(Point(int(np.floor(p.x)), int(np.ceil(p.y))))
    px.add(Point(int(np.ceil(p.x)), int(np.floor(p.y))))
    px.add(Point(int(np.ceil(p.x)), int(np.ceil(p.y))))
    return px


def direction(p1, p2):
    dir_x = (p2.x - p1.x) / np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
    dir_y = (p2.y - p1.y) / np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
    return dir_x, dir_y


def bubble_from_max(loc_max, img, len_curve_threshold=30):
    """
    TODO: refactor to bubble_by_grad
    """
    curr_pos = loc_max
    curve = [curr_pos]
    while len(curve) < len_curve_threshold:

        try:
            neighbours = get_lower_right_neighbours(curr_pos)
            grad = np.asarray([img[curr_pos.get_coord()] - img[neigh.get_coord()]
                               for neigh in neighbours])
            next_pos = neighbours[grad.argmin()]
            curr_pos = deepcopy(next_pos)
            curve += [next_pos]

        except IndexError:
            print("WARNING: boundary of image reached!")
            break

    bubble_radius = curve[-1].y - loc_max.y
    bubble_center = Point(loc_max.x, curve[-1].y)

    return curve, Circle(bubble_center.x, bubble_center.y, bubble_radius)


def bubble_by_zeroing(loc_max, input_img, len_curve_threshold=120,
                      proportion_threshold=3., neighbours_part="right"):
    # TODO: rename to curve_by_max
    """
    :param loc_max: location of maximum of type Point.
    :param input_img: input image numpy array.
    :param len_curve_threshold: maximum length of curve.
    :param proportion_threshold: Width over height of curve.
    :param neighbours_part: right, left, upper, lower, lower_right, upper_right or all.
    :return: curve describing bubble.
    """
    img = input_img
    curr_pos = loc_max
    curve = [curr_pos]
    extr_x = curr_pos.x
    while len(curve) < len_curve_threshold:
        try:
            neighbours = get_neighbours(curr_pos, part=neighbours_part)
            next_pos_arg = np.asarray([img[neigh.get_coord()]
                                       for neigh in neighbours]).argmax()
            next_pos = neighbours[next_pos_arg]
            curve += [next_pos]
            curr_pos = next_pos

            # Abbruchbedingung
            if neighbours_part == "lower_right":
                if curr_pos.x > extr_x:
                    extr_x = curr_pos.x
            elif neighbours_part == "upper_right":
                if curr_pos.x <= extr_x:
                    extr_x = curr_pos.x

            if (extr_x - curve[0].x) == 0:
                pass
            elif float(np.abs((curr_pos.y - curve[0].y)/(extr_x - curve[0].x))) \
                    >= proportion_threshold:
                break
        except IndexError:
            print("WARNING: reached image boundary!")
            break
            
    return curve


def my_digitize(x, bins):
    # check input is correct, i.e.
    # bins is monotonic and equidistant

    bin_size = (bins[1] - bins[0])

    if type(x) == type(np.empty((0, 0))):
        dig_x = np.empty(x.shape)
        if len(dig_x.shape) == 2:
            for i, row in enumerate(x):
                for j, val in enumerate(row):
                    for b in bins:
                        if b - bin_size/2 <= val < b + bin_size/2:
                            dig_x[i, j] = b
            return dig_x

        elif len(dig_x.shape) == 1:
            for i, val in enumerate(x):
                for b in bins:
                    if b - bin_size/2 <= val < b + bin_size/2:
                        dig_x[i] = b
            return dig_x
    else:
        if x >= bins[-1] + bin_size / 2:
            return bins[-1]
        elif x < bins[0] - bin_size / 2:
            return bins[0]
        else:
            for idx, b in enumerate(bins):
                print("xxxx interval starts", idx)
                print("b - bin_size / 2 = ", b - bin_size / 2)
                print("b + bin_size / 2 = ", b + bin_size / 2)
                print("xxxxx interval ends", idx)
                if b - bin_size / 2 <= x < b + bin_size / 2:
                    return b
            print("x doesn't fit in bin !! x = ", x)


def orientation_from_image(img, smooth_mask, digitize=False):

    Axx, Axy, Ayy = utils.my_structure_tensor(img, smooth_mask)
    angle = .5 * np.arctan2(2 * Axy, Ayy - Axx)
    if digitize:
        bins = np.array([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
        return my_digitize(angle, bins)
    else:
        return angle


def neighbour_from_orientation(curr_pos, curr_orientation, curve_preferred_direction):
    if curr_orientation == -np.pi / 2 or curr_orientation == np.pi / 2:
        return Point(curr_pos.x, curr_pos.y + 1)
    elif curr_orientation == np.pi / 4:
        return Point(curr_pos.x - 1, curr_pos.y + 1)
    elif curr_orientation == 0:
        if curve_preferred_direction == "UP":
            return Point(curr_pos.x - 1, curr_pos.y)
        elif curve_preferred_direction == "DOWN":
            return Point(curr_pos.x + 1, curr_pos.y)
    elif curr_orientation == -np.pi / 4:
        return Point(curr_pos.x + 1, curr_pos.y + 1)
    else:
        print("WARNING: angle case not covered, angle=", curr_orientation)
        return curr_pos


def is_complete(curve,
                extr_x,
                curr_pos,
                proportion_threshold):

    if (extr_x - curve[0].x) == 0:
        pass
    elif float(np.abs((curr_pos.y - curve[0].y) / (extr_x - curve[0].x))) \
            >= proportion_threshold:
        return True
    else:
        return False


def curve_from_orientation(input_img,
                           loc_max,
                           smooth_mask,
                           len_curve_threshold=120,
                           proportion_threshold=3,
                           curve_preferred_direction="UP",
                           include_first_point=True):
    img = input_img
    curr_pos = loc_max
    curve = [curr_pos]
    extr_x = curr_pos.x
    orient = orientation_from_image(img, smooth_mask, digitize=True)

    while len(curve) < len_curve_threshold:
        try:
            curr_orientation = orient[curr_pos.get_coord()]
        except IndexError:
            print("Image boundary reached! current position:", curr_pos.get_coord())
            break

        next_pos = neighbour_from_orientation(curr_pos,
                                              curr_orientation,
                                              curve_preferred_direction)
        curve += [next_pos]
        curr_pos = next_pos

        # Abbruchbedingung
        if curve_preferred_direction == "DOWN":
            if curr_pos.x > extr_x:
                extr_x = deepcopy(curr_pos.x)
        else:
            if curr_pos.x < extr_x:
                extr_x = deepcopy(curr_pos.x)

        if is_complete(curve, extr_x, curr_pos, proportion_threshold):
            break

    if include_first_point:
        return curve
    else:
        return curve[1:]


def fit_along_line(samples, values, fit_line):
    """
    :param samples: list of 2d points of type Point.
    :param values: gray values of type Number.
    :param fit_line:
    :return: gaussian mean from fit of type Point.
    """
    y_data = values
    x_data = np.arange(0, len(y_data), 1)
    lin_dir = fit_line.direction
    fit_converged = False
    try:
        (_, mu, _), _ = curve_fit(gauss, x_data, y_data)
        fit_converged = True
        print("fit converged! mu=", mu)
        print("result point = ", (mu * lin_dir + samples[0]).get_coord())
    except RuntimeError:
        mu = x_data[int(np.floor(len(x_data)/2))]
        fit_converged = False
        print("no fit, mu =", mu)
        print("no fit, direction=", lin_dir.get_coord())

    print("samples 0:", samples[0].get_coord())
    print("len(samples)", len(samples))
    mu_arg = int(np.floor(mu))
    if mu_arg >= len(samples):
        result = samples[-1]
    elif mu_arg < 0:
        result = samples[0]
    else:
        result = samples[int(np.floor(mu))]
    if fit_converged:
        result.is_fit_result = True
    print("result point: ", result.get_coord())
    return result


def sample_fit_line(fit_line, pos, img, max_len, nb_samples, bsize):
    """
    This is where the magic happens...
    :param fit_line: all fit points lie on this line of type Line
    :param pos: maximum length of line will be computed starting
    from this point of type Point.
    :param img: 2d np array
    :param max_len: maximum length of fit segment
    :param nb_samples: number of samples on the fit line
    :param bsize: size of bounding box per sample
    :return: samples, values. Both are lists of same size. samples is a
    list of Point(s), and values is a list of Number(s).
    """
    sample_step_size = float(max_len) / nb_samples
    curr_pos = pos - float(max_len)/2 * fit_line.direction
    samples = []
    values = []

    while len(values) < nb_samples:
        #try:
            val = 0
            curr_box = Rectangle()
            curr_box.by_center_width_height(curr_pos.x, curr_pos.y, bsize, bsize)
            neighbours = get_neighbouring_pixels(curr_pos)
            nb_neighbours = len(neighbours)
            for neigh in neighbours:
                if neigh.has_negative_coord():
                    print("-- neighbour has negative coordinate")
                    pass
                neigh_box = Rectangle()
                neigh_box.by_center_width_height(neigh.x, neigh.y, 1, 1)
                weight = inter_area(neigh_box, curr_box)
                val += weight * img[neigh.get_coord()] / nb_neighbours
            values += [val]
            samples += [curr_pos]

            curr_pos = curr_pos + sample_step_size * fit_line.direction

        #except IndexError:
        #   print("point outside image border, pass...", curr_pos.get_coord())
        #   curr_pos = curr_pos + sample_step_size * fit_line.direction
        #   break

    assert len(values) == len(samples)
    return samples, values


def curve_from_orientation_fit(input_img,
                               start_point,
                               smooth_mask,
                               len_curve_threshold=120,
                               proportion_threshold=3,
                               curve_preferred_direction="UP",
                               include_first_point=True,
                               fit_line_max_length=10,
                               fit_line_nb_samples=20,
                               fit_line_sample_box_size=1,
                               next_step_size=1/np.sqrt(2),
                               verbose=False):
    """
    Given an image containing a bubble and a starting point (typically a local maximum),
    curve_from_orientation returns a list of Points, i.e. curve, along the curvature of
    the bubble.

    :param input_img: input image as np.array
    :param start_point: starting point as Point. Typically local maximum
    :param smooth_mask: smoothing mask of type np.array used by structure tensor to produce orientation.
    :param len_curve_threshold: maximum curve length
    :param proportion_threshold: curve width / curve height. When this threshold is reached,
    the curve is complete
    :param curve_preferred_direction: "UP" or "DOWN". Decides where to choose the next point when
    orientation is ambiguous.
    :param include_first_point: if True start_point is included in the result
    :param fit_line_max_length: maximum fit line length
    :param fit_line_nb_samples: number of samples on the fit line
    :param fit_line_sample_box_size: fit line box size
    :param next_step_size: distance between current point and next starting point
    :para, verbose: if True print progress and log info
    :return: list of Point describing the outer bubble curvature.

    Example usage:
    curve = curve_from_orientation_fit(input_img=one_pad,
                                   loc_max=loc_max_pad,
                                   smooth_mask=smooth_mask,
                                   next_step_size=1,
                                  proportion_threshold=2,
                                   fit_line_nb_samples=30,
                                   len_curve_threshold=40,
                                   curve_preferred_direction="DOWN",
                                  include_first_point=True)
    curve += curve_from_orientation_fit(input_img=one_pad,
                               loc_max=loc_max_pad,
                               smooth_mask=smooth_mask,
                               next_step_size=1,
                              proportion_threshold=2,
                                len_curve_threshold=13,
                                fit_line_max_length=6,
                            fit_line_nb_samples=30,
                              curve_preferred_direction="UP",
                              include_first_point=False)
    """
    img = input_img
    curr_pos = start_point
    curve = [curr_pos]
    extr_x = curr_pos.x
    max_len  = fit_line_max_length
    nb_samples  = fit_line_nb_samples
    bsize = fit_line_sample_box_size
    ssize = next_step_size
    orient = orientation_from_image(img, smooth_mask, digitize=False)

    for _ in range(len_curve_threshold):
        try:
            curr_orientation = orient[curr_pos.get_coord(dtype="int")]
            curr_orient_dir = Point(np.sin(curr_orientation), np.cos(curr_orientation))
            fit_line = Line(point=curr_pos, direction=curr_orient_dir)
            samples, values = sample_fit_line(fit_line, curr_pos, img, max_len, nb_samples, bsize)
            refined_curr_pos = fit_along_line(samples, values, fit_line)

            ###########################################################################
                                        #####  log  ######
            if verbose:
                print("############ new position ############")
                print("curr_pos: ", curr_pos.get_coord())
                print("curr_orientation", curr_orientation / (2*np.pi) * 360)
                print("fit_line=", fit_line.point.get_coord(), fit_line.direction.get_coord())
                for s, v in zip(samples, values):
                    print(s.get_coord(), v)
                if isinstance(refined_curr_pos, Point):
                    print("refined_curr_pos=", refined_curr_pos.get_coord())
                print("current curve len", len(curve))

            ############################################################################

            if refined_curr_pos.is_fit_result:
                curve += [refined_curr_pos]

            if curve_preferred_direction == "DOWN":
                next_pos_orient = curr_orientation + np.pi/2
            else: # curve_preferred_direction == "UP":
                next_pos_orient = curr_orientation - np.pi / 2
            next_pos_dir = Point(np.sin(next_pos_orient), np.cos(next_pos_orient))
            next_pos = curr_pos + ssize * next_pos_dir
            curr_pos = next_pos

            # Abbruchbedingung
            if curve_preferred_direction == "DOWN":
                if curr_pos.x > extr_x:
                    extr_x = deepcopy(curr_pos.x)
            else: # curve_preferred_direction == "UP":
                if curr_pos.x < extr_x:
                    extr_x = deepcopy(curr_pos.x)
            if is_complete(curve, extr_x, curr_pos, proportion_threshold):
                break

        except IndexError:
            print("Image boundary reached! current position:", curr_pos.get_coord())
            break

    if include_first_point:
        return curve
    else:
        return curve[1:]


def bubble_from_curve(curve, output_shape="circle"):
    loc_max = curve[0]
    max_point = loc_max
    for c in curve:
        if c.x > max_point.x:
            max_point = c
    bubble_circ = Circle(loc_max.x, max_point.y, (max_point.x - loc_max.x))
    if output_shape == "circle":
        return bubble_circ
    else: # shape == "rectangle":
        return circle_to_rectangle(bubble_circ)


def _lin_func(x):
    return 1 * x + 1.5


def half_circ(x, a, b, r):
    return np.sqrt(np.abs(r**2 - (x-a)**2)) + b**2


def dist(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def circle_fit(x_data, y_data):
    """
    code copied from https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
    TODO: adjust to own code style
    (---- ^ lol no one has time for that)
    :param x_data:
    :param y_data:
    :return:
    """
    x = x_data
    y = y_data
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the
        mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center_2, ier = leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2 = calc_R(*center_2)
    R_2 = Ri_2.mean()
    residu_2 = sum((Ri_2 - R_2) ** 2)

    return xc_2, yc_2, R_2, residu_2


def bubble_from_curve_fit(curve, output_shape="circle", verbose=False):

    x_data = [pt.x for pt in curve]
    y_data = [pt.y for pt in curve]

    xc, yc, r, _ = circle_fit(x_data, y_data)

    pred_circ = Circle(xc, yc, r)

    if verbose:
        ################
        print("----- bubble_from_curve info -----")
        print("x_data", x_data)
        print("y_data", y_data)
        print("xc, yc, r", xc, yc, r)
        ################

    if output_shape == "rectangle":
        return circle_to_rectangle(pred_circ)
    else: # output_shape == "circle":
        return pred_circ


def _slope(curve, idx):
    if len(curve) - 1 > idx > 0:
        y1 = curve[idx-1].y
        x1 = curve[idx-1].x
        y2 = curve[idx+1].y
        x2 = curve[idx+1].x
    else:
        return np.nan

    if y2 != y1:
        return float(x2 - x1)/(y2 - y1)
    else:
        if x2 > x1:
            return +np.inf
        else:
            return -np.inf


def _angle_from_slope(sl):
    if np.isnan(sl):
        return np.nan
    else:
        assert np.pi / 2 >= np.arctan(sl) >= -np.pi / 2
        return np.arctan(sl)


def _refine_pos(slope, pt, img):
    angle = _angle_from_slope(slope)
    if np.isnan(angle):
        return pt
    bins = np.array([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
    dig_angle_arg = np.digitize(angle, bins)
    if bins[dig_angle_arg - 1] == bins[0] or dig_angle_arg == bins[-1]:
        fit_line = [Point(pt.x, pt.y-1), pt, Point(pt.x + 1, pt.y+1)]
        ex = Point(0, 0)
        ey = Point(0, 1)

    elif bins[dig_angle_arg - 1] == bins[-1]:
        # fit_line = [Point(pt.x - 1, pt.y), pt, Point(pt.x + 1, pt.y)]
        fit_line = [Point(pt.x, pt.y - 1), pt, Point(pt.x + 1, pt.y + 1)]
        ex = Point(0, 0)
        ey = Point(0, 1)
    elif bins[dig_angle_arg - 1] == bins[1]:
        fit_line = [Point(pt.x-1, pt.y-1), pt, Point(pt.x+1, pt.y+1)]
        ex = Point(1, 0)
        ey = Point(0, 1)
    elif bins[dig_angle_arg - 1] == bins[2]:
        fit_line = [Point(pt.x, pt.y-1), pt, Point(pt.x, pt.y+1)]
        ex = Point(0, 0)
        ey = Point(0, 1)
    elif bins[dig_angle_arg - 1] == bins[3]:
        fit_line = [Point(pt.x+1, pt.y-1), pt, Point(pt.x-1, pt.y+1)]
        ex = Point(-1, 0)
        ey = Point(0, 1)
    else:
        return pt

    try:
        x_data = [0, 1, 2]
        y_data = [img[c.x, c.y] for c in fit_line]
        (_, mu, _), _ = curve_fit(gauss, x_data, y_data)
        if mu > max(x_data) or mu < min(x_data):
            return pt
    except RuntimeError:
        mu = x_data[1]
    except IndexError:
        return pt
    refined_pt_x = mu * (ex.x + ey.x) + fit_line[0].x
    refined_pt_y = mu * (ex.y + ey.y) + fit_line[0].y

    return Point(refined_pt_x, refined_pt_y)


def _refine_pos_max(slope, pt, img, fit_line_size=5):
    angle = _angle_from_slope(slope)
    if np.isnan(angle):
        print("------------- ##### angle is nan ######-------")
        return pt
    bins = np.array([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])
    angle = my_digitize(angle, bins)

    if angle == -np.pi/2 or angle == np.pi/2:
        j_arr = np.arange(-np.round(fit_line_size/2), np.round(fit_line_size/2), dtype=int)
        i_arr = np.zeros(j_arr.shape, dtype=int)
    elif angle == -np.pi/4:
        # fit_line = [Point(pt.x-1, pt.y - 1), pt, Point(pt.x+1, pt.y + 1)]
        i_arr = np.arange(-np.round(fit_line_size / 2), np.round(fit_line_size / 2), dtype=int)
        j_arr = np.arange(-np.round(fit_line_size / 2), np.round(fit_line_size / 2), dtype=int)
    elif angle == 0:
        # fit_line = [Point(pt.x-1, pt.y), pt, Point(pt.x+1, pt.y)]
        i_arr = np.arange(-np.round(fit_line_size / 2), np.round(fit_line_size / 2), dtype=int)
        j_arr = np.zeros(i_arr.shape, dtype=int)
    elif angle == np.pi/4:
        # fit_line = [Point(pt.x-1, pt.y + 1), pt, Point(pt.x+1, pt.y - 1)]
        i_arr = np.arange(-np.round(fit_line_size / 2), np.round(fit_line_size / 2), dtype=int)
        j_arr = np.flip(i_arr, axis=0)

    fit_line = [Point(pt.x - i, pt.y + j) for (i, j) in zip(i_arr, j_arr)]

    max_val = 0
    max_idx = 0
    for idx, p in enumerate(fit_line):
        print("p.x, p.y", p.x, p.y)
        try:
            if img[p.x, p.y] > max_val:
                max_val = img[p.x, p.y]
                max_idx = idx
        except IndexError:
            pass

    print("old pt", pt.get_coord())
    print("new pt", fit_line[max_idx].get_coord())
    return fit_line[max_idx]


def curve_refine(curve, img, mode="fit"):
    curr_curve = deepcopy(curve)
    smooth_curve = []
    sharp_curve = []
    for idx, c in enumerate(curr_curve):
        sl = _slope(curr_curve, idx)
        if mode == "fit":
            pt = _refine_pos(sl, c, img)
        elif mode == "max":
            pt = _refine_pos_max(sl, c, img)
        sharp_curve.append(Point(np.round(pt.x), np.round(pt.y)))
        smooth_curve.append(pt)
    return smooth_curve


def bubble_from_gauss_max(input_img, bin_thr, p0=None):
    _, bin_img = threshold(input_img, bin_thr, 255, THRESH_BINARY)
    curve = [Point(0, 0)] * bin_img.shape[0]
    for row in range(bin_img.shape[0]):
        signal_y = bin_img[row, :]
        signal_x = np.arange(len(signal_y))
        if sum(signal_y) > 0:
            (_, mu, _), _ = curve_fit(gauss, signal_x, signal_y, p0=p0)
            curve[row] = Point(row, mu)
    return curve


def x_train_from_curves(curves, nb_features):
    x_train = []
    for idx, cur in enumerate(curves):
        x_curve = []
        for pt in cur:
            x_curve.append(pt.x)
        x_train.append(x_curve)
    return [resize_1d(x_tr, nb_features) for x_tr in x_train]


def binarize(img, thr, val_min=0, val_max=1):
    for i, row in enumerate(img):
        for j, col in enumerate(row):
            if img[i, j] >= thr:
                img[i, j] = val_max
            else:
                img[i, j] = val_min
    return img


def adaptive_threshold(img_input, window_size, param, val_min=0, val_max=1):
    img_result = deepcopy(img_input)
    for i, row in enumerate(img_input):
        for j, col in enumerate(row):

            i_min = i-window_size
            i_max = i+window_size
            j_min = j-window_size
            j_max = j+window_size

            # no periodic boundary conditions
            if i-window_size < 0:
                i_min = 0
            if i+window_size > img_input.shape[0]:
                i_max = i
            if j-window_size < 0:
                j_min = 0
            if j+window_size > img_input.shape[1]:
                j_max = j

            region = img_input[i_min:i_max, j_min:j_max]
            thr = np.mean(region) - param

            # binarize
            if img_input[i, j] >= thr:
                img_result[i, j] = val_max
            else:
                img_result[i, j] = val_min
    return img_result


def im_center(im):
    if im.shape[0] % 2 == 0:
        return Point(np.floor(im.shape[0] / 2), np.ceil(im.shape[1] / 2))
    else:
        return Point(np.ceil(im.shape[0]/2), np.ceil(im.shape[1]/2))


def _get_structure_tensor_features(im, features_window_size, sigma_st):
    features_arr = np.empty(0)
    assert im.shape[0] == im.shape[1], "image must have a square shape"
    center = im_center(im)
    Axx, Axy, Ayy = feature.structure_tensor(im, sigma=sigma_st)
    l1, l2 = feature.structure_tensor_eigvals(Axx, Axy, Ayy)
    l1 = utils.extract_pad_image(input_img=l1, pt=center, window_size=features_window_size)
    l2 = utils.extract_pad_image(input_img=l2, pt=center, window_size=features_window_size)
    l1 = arr_to_vec(l1)
    l2 = arr_to_vec(l2)
    features_arr = np.append(features_arr, l1)
    features_arr = np.append(features_arr, l2)
    return features_arr


def _get_hessian_matrix_features(im, features_window_size, sigma_hm):
    features_arr = np.empty(0)
    center = im_center(im)
    Hxx, Hxy, Hyy = feature.hessian_matrix(im, sigma=sigma_hm)
    h1, h2 = feature.hessian_matrix_eigvals(Hxx, Hxy, Hyy)
    h1 = utils.extract_pad_image(input_img=h1, pt=center, window_size=features_window_size)
    h2 = utils.extract_pad_image(input_img=h2, pt=center, window_size=features_window_size)
    h1 = arr_to_vec(h1)
    h2 = arr_to_vec(h2)
    features_arr = np.append(features_arr, h1)
    features_arr = np.append(features_arr, h2)
    return features_arr


def get_max_features(img,
                     features_window_size,
                     sigma_structure_tensor=list([1]),
                     sigma_hessian_matrix=list([1])):

    features_arr = np.empty(0)
    im = img / img.max()

    for sigma_st in sigma_structure_tensor:
        feat_st = _get_structure_tensor_features(im, features_window_size, sigma_st)
        features_arr = np.append(features_arr, feat_st)

    for sigma_hm in sigma_hessian_matrix:
        feat_hm = _get_hessian_matrix_features(im, features_window_size, sigma_hm)
        features_arr = np.append(features_arr, feat_hm)

    return features_arr


def bubbles_from_image(img,
                       classifier_filename,
                       error_func,
                       from_fit=False,
                       threshold_abs=60,
                       min_distance=5):
    lm = peak_local_max(img,
                        threshold_abs=threshold_abs, min_distance=min_distance)
    local_max_candidates = [Point(l[0], l[1]) for l in lm]

    # get features for classification
    local_max_candidates_img = [utils.extract_pad_image(input_img=img, pt=lmd, window_size=10) for lmd in
                                local_max_candidates]
    local_max_candidates_features = [get_max_features(img=im, features_window_size=3) for im in
                                     local_max_candidates_img]
    with open(classifier_filename, "rb") as handle:
        clf = pickle.load(handle)
    lm_bubble = clf.predict(local_max_candidates_features)

    # keep maxima classified as bubbles
    local_max_valid = []
    for lmd, lmb in zip(local_max_candidates, lm_bubble):
        if lmb == 1:
            local_max_valid.append(lmd)

    curves = []
    for lm in local_max_valid:
        cu = bubble_by_zeroing(input_img=img,
                               loc_max=lm,
                               neighbours_part="upper_right",
                               len_curve_threshold=40,
                               proportion_threshold=2.2)
        cu += bubble_by_zeroing(input_img=img,
                                loc_max=lm,
                                neighbours_part="lower_right",
                                len_curve_threshold=40,
                                proportion_threshold=2.2)
        curves.append(cu)

    if from_fit:
        return [bubble_from_curve_fit(curve=cu) for cu in curves]
    else:
        return [bubble_from_curve(cu) for cu in curves]


def circle_from_vertical_profile(vert_sig, sigma=1, widths=[4]):
    idx_start = 0
    idx_end = 2

    vert_sig = gaussian_filter1d(vert_sig, sigma)
    peaks_arg = find_peaks_cwt(vert_sig, widths)

    if len(peaks_arg) >= 3:
        idx_start = 1
        idx_end = 3
    elif len(peaks_arg) == 2:
        idx_start = 0
        idx_end = 2
    elif len(peaks_arg) < 2:
        return Circle(0, 0, 0)
    upper_edge, lower_edge = peaks_arg[np.argsort(vert_sig[peaks_arg])][idx_start:idx_end]

    radius = float(upper_edge - lower_edge) / 2
    cent_x = float(upper_edge + lower_edge) / 2
    return Circle(cent_x, np.nan, np.abs(radius))


def green_bubble_one(subimage,
                     method,
                     hough_radii=None,
                     total_num_peaks=10,
                     max_offset=10,
                     fit_refine=False,
                     sigma=3,
                     verbose=False):
    """
    estimate bubble radius and position using Hough circle transform,
    assuming there is only one backlit bubble in subimage located around
    the image center.
    :param subimage: input image containing one bubble.
    :param method: "hough", "peak_dist" or "vertical_profile"
    :param hough_radii: list of radii candidates
    :param total_num_peaks: number of candidate bubbles
    :param max_offset: maximum distance between image center and circle center.
    :param sigma: if method == vertical_profile, smooth vertical profile with
            gaussian filter
    :param fit_refine: for method="peak_dist" only. If True refine position of center
            and position of edge using an analytical gauss fit.
    :return: Circle describing location and radius of bubble.
    """
    edges = sobel(subimage)
    im_cent = Point(int(subimage.shape[1]/2) , int(subimage.shape[0]/2))

    if method == "hough":
        hough_res = hough_circle(edges, hough_radii)
        _, center_x, center_y, radii = hough_circle_peaks(hough_res,
                                                   hough_radii,
                                                   total_num_peaks=total_num_peaks)
        res_circles = [Circle(cx, cy, r) for cx, cy, r in zip(center_x, center_y, radii)]
        if len(res_circles) != 0:
            avg_x = np.mean([circ.x for circ in res_circles if circ.x - im_cent.x < max_offset])
            avg_y = np.mean([circ.y for circ in res_circles if circ.y - im_cent.y < max_offset])
            avg_r = np.mean([circ.radius for circ in res_circles])
        else:
            avg_x = np.nan
            avg_y = np.nan
            avg_r = np.nan

        return Circle(avg_x, avg_y, avg_r)

    elif method == "peak_dist":

        plm = peak_local_max(subimage, min_distance=5)
        local_max_candidates = [Point(l[0], l[1]) for l in plm]
        mid_peak = Point(im_cent.x, im_cent.y)

        for mo in range(max_offset):
            lm = [l for l in local_max_candidates if dist(l, im_cent) <= mo+1]
            if verbose:
                print("current image center candidate", [p.get_coord() for p in lm])
            if len(lm) == 1:
                mid_peak = lm[0]
                break

        signal = gaussian_filter1d(edges[mid_peak.y, :], sigma=sigma)
        max_arg = argrelextrema(signal, np.greater)[0]

        candidate_right_peak_args = [y - mid_peak.y for y in max_arg if y - mid_peak.y > 0]
        if len(candidate_right_peak_args) > 0:
            min_dist = min(candidate_right_peak_args)
            print("min distance to image center", min_dist)
        else:
            return Circle(1,1,1)
        if verbose:
            print("max_arg", max_arg)
            print("len max_arg", len(max_arg))
            print("signal local maxima:", [signal[arg] for arg in max_arg])
        right_peak_arg = [arg for arg in max_arg if np.abs(arg-mid_peak.y) == min_dist][0]
        right_peak = Point(right_peak_arg, mid_peak.y)

        if fit_refine:
            g0 = subimage[mid_peak.y, mid_peak.x - 1]
            g1 = subimage[mid_peak.y, mid_peak.x]
            g2 = subimage[mid_peak.y, mid_peak.x + 1]

            mu_right_peak_x, _ = utils.gauss_fit_analytical(g0, g1, g2)
            right_peak.x = mu_right_peak_x + right_peak.x
            if verbose:
                print("mu_right_peak_x", mu_right_peak_x)
                print("right_peak:", right_peak.get_coord())

        radius = dist(mid_peak, right_peak)
        if verbose:
            print("radius =", radius)

        return Circle(mid_peak.x, mid_peak.y, radius)

    elif method == "vertical_profile":
        max_circle =  Circle(1, 1, 1)
        edges = scharr(subimage)
        cent = im_center(subimage)
        for idx in np.arange(int(cent.y - max_offset/2), int(cent.y + max_offset/2), 1):
            vert_prof = edges[:, idx]
            curr_circle = circle_from_vertical_profile(vert_prof, sigma=sigma)
            curr_radius = curr_circle.radius
            if verbose:
                print("current radius:", curr_radius)
            if curr_radius > max_circle.radius:
                max_circle.radius = curr_radius
                max_circle.y = idx
                max_circle.x = curr_circle.x
        return max_circle

    else:
        raise ValueError("method " + method + " is not supported")


def red_bubble_one(subimg, method, smooth_mask=None, verbose=True):

    plm = peak_local_max(subimg, min_distance=5, threshold_abs=50)

    if verbose:
        print("#-#-#-#-#-#- new_bubble -#-#-#-#-#-#-", plm)

    if len(plm) != 0:
        plm = plm[0]
        lm = Point(plm[0], plm[1])
    else:
        return [Circle(1, 1, 1)]

    if method == "circle_fit":
        curve = curve_from_orientation_fit(input_img=subimg,
                                           start_point=lm,
                                           smooth_mask=smooth_mask,
                                           len_curve_threshold=30,
                                           proportion_threshold=4,
                                           curve_preferred_direction="UP",
                                           include_first_point=True,
                                           fit_line_max_length=5,
                                           fit_line_nb_samples=5,
                                           fit_line_sample_box_size=1,
                                           verbose=verbose)

        curve += curve_from_orientation_fit(input_img=subimg,
                                           start_point=lm,
                                           smooth_mask=smooth_mask,
                                           len_curve_threshold=30,
                                           proportion_threshold=4,
                                           curve_preferred_direction="DOWN",
                                           include_first_point=False,
                                           fit_line_max_length=5,
                                           fit_line_nb_samples=10,
                                           fit_line_sample_box_size=1,
                                           verbose=verbose)
        if verbose:
            print("################################### curve", curve)

        return bubble_from_curve_fit(curve)

    elif method == "peak_dist":
        pred_circ = detec_bubble(subimg,
                            calib_radius_func=lambda x: x,
                            threshold_abs=50,
                            min_distance=10,
                            classifier="bubble_forever",
                            output_shape="Circle",
                            signal_len=40,
                            flip_signal=True,
                            verbose=verbose)
        mean_radii = np.mean([pr.radius for pr in pred_circ])
        valid_bubbles = [prc for prc in pred_circ
                         if prc.radius >= mean_radii]# and len(pred_circ) > 1]
        return valid_bubbles

    else:
        print("method '{0}' not supported".format(method))


def main():

    # read image(s)
    filename = "/Users/Habib/Google Drive/Uni Heidelberg/12 SS 2018/" \
               "Masterarbeit/gradient_ansatz/data/one_mess.png"
    img = imread(filename, 0)

    #### mask ###
    size = 31
    xx = np.linspace(0, 10, size)
    yy = np.linspace(0, 10, size)
    XX, YY = np.meshgrid(xx, yy)
    smooth_mask = utils.gauss_2d_mask([XX, YY], amp=10, mu=[5, 5], sigma=[1, 1])

    #### local maximum ####
    lm = peak_local_max(img, threshold_abs=100)[0]
    loc_max = Point(lm[0], lm[1])

    curve = curve_from_orientation_fit(input_img=img,
                                       start_point=loc_max,
                                       smooth_mask=smooth_mask)


if __name__ == "__main__":
    main()

