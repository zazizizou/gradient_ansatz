import numpy as np
from scipy.ndimage.filters import convolve
from skimage.filters import scharr_v, scharr_h
import re
from copy import deepcopy
from scipy.misc import imresize
import shapes


"""
Coordinate system is:
.-–-> y
|
|
v
x
"""


def pixel_coord_to_box(pt):

    xmin = pt.x - 0.5
    ymin = pt.y - 0.5
    xmax = pt.x + 0.5
    ymax = pt.x + 0.5
    return shapes.Rectangle(xmin, ymin, xmax, ymax)


def dot_prod(p1, p2):
    return (p1.x * p2.x) + (p1.y * p2.y)


def divide_img_grid(img, nx, ny):
    """
    produce (nx * ny) images from img.
    Returns a list of numpy arrays (sub-images)
    """
    sub_img = []
    size_x = int(img.shape[0] / nx)
    size_y = int(img.shape[1] / ny)
    
    for i in range(nx):
        for j in range(ny):
            x_start = i     * size_x
            x_end   = (i+1) * size_x
            y_start = j     * size_y
            y_end   = (j+1) * size_y
            
            sub_img += [img[x_start:x_end, y_start:y_end]]
            
    return sub_img


def ll_to_ul(coord, img_size):
    """
    change coordinate system from lower left (ll) to upper left (ul)
    """
    x_axis = img_size/2
    if coord < x_axis:
        return coord + abs(2*(coord-x_axis))
    else:
        return coord - abs(2*(coord-x_axis))


def add_motion_blur(img, step=1/2):
    img = img.astype("float64")
    motion_blur_filter = np.array([[0,0,0],[step, step, step],[0,0,0]])
    return convolve(img, motion_blur_filter)


def add_gauss_noise(img, sigma, mu, ampl, normalize_output=True):
    img = img.astype('float64')
    noise = np.random.randn(img.shape)
    gauss_noise = sigma * noise + mu
    gauss_noise = gauss_noise / np.max(gauss_noise) * ampl
    noisy_img = gauss_noise + img

    if normalize_output:
        return noisy_img + np.min(noisy_img) / np.max(noisy_img)
    else:
        return noisy_img + np.min(noisy_img) / np.max(noisy_img) * 255


def add_background_from_measurement(render, background, normalize_output):
    render = render.astype("float64")
    background = background.astype("float64")
    if normalize_output:
        return (render + background) / np.max(render + background)
    else:
        return (render + background) / np.max(render + background) * 255


def intersection_over_union(boxA, boxB):
    """
    computes intersection over union: (Area of Overlap) / (Area of Union)
    boxA = (x,y w, h), where (x,y) is the upper left corner of the 
    rectangle and w and h are with and height of the rectangle
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    # (origin in upper left)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = (xB - xA + 1) * (yB - yA + 1)
    boxAArea = (boxA[2]+1) * (boxA[3]+1)
    boxBArea = (boxB[2]+1) * (boxB[3]+1)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    if iou <= 0:
        return 0
    else:
        return iou


def aspect_ratio(box, epsilon=1e-5):
    """
    determines aspect ratio of a rectangle box (x, y, w, h), 
    where (x,y) is the upper left corner of the rectangle and
    w and h are with and height of the rectangle.
    """
    epsilon = epsilon
    w = box[2]
    h = box[3]

    if h != 0:
        return w / h
    else:
        return w / epsilon


def find_re(reg, text):
    m = re.search(reg, text)
    return m.group(0)


def resize_1d(signal, nb_points, interp="bilinear"):
    signal = np.asarray(signal)
    result = np.reshape(signal, (1, len(signal)))
    result = imresize(result, (1, nb_points), interp=interp)
    return result.reshape((nb_points))


def gauss(x, a, mu, sigma):
    return a * np.exp(- (x - mu)**2/sigma**2)


def gauss_2d_mask(x, sigma, amp=10, mu=[0, 0], padding=0):
    return amp/(2*np.pi*sigma[0]*sigma[1]) * \
           np.exp(-(x[0] - mu[0])**2/(2*sigma[0]**2) - (x[1] - mu[1])**2/(2*sigma[1]**2))


def my_structure_tensor(img, smooth_mask):
    """
    Structure tensor is defined as
    [ Axx   Axy ]
    [ Axy   Axx ]
    computes derivative with Scharr filter and smooths with given
    smooth_mask.
    """
    imx = scharr_h(img)
    imy = scharr_v(img)

    Axx = convolve(imx * imx, smooth_mask)
    Axy = convolve(imx * imy, smooth_mask)
    Ayy = convolve(imy * imy, smooth_mask)

    return Axx, Axy, Ayy


def extract_pad_image(input_img, pt, window_size, pad_mode="edge"):
    im = deepcopy(input_img)
    x_start = pt.x - window_size
    y_start = pt.y - window_size
    x_end   = pt.x + window_size
    y_end   = pt.y + window_size

    x_min_pad_val = 0
    x_max_pad_val = 0
    y_min_pad_val = 0
    y_max_pad_val = 0

    if x_start < 0:
        x_min_pad_val = np.abs(x_start)
    if x_end >= input_img.shape[0]:
        x_max_pad_val = x_end - input_img.shape[0] - 1
    if y_start < 0:
        y_min_pad_val = np.abs(y_start)
    if y_end >= input_img.shape[1]:
        y_max_pad_val = y_end - input_img.shape[1] - 1

    im = np.pad(im,
                ((x_min_pad_val, x_max_pad_val),
                     (y_min_pad_val, y_max_pad_val)),
                pad_mode)

    return im[x_start:x_end, y_start:y_end]


