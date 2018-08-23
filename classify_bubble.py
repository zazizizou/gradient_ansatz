import numpy as np
from skimage.feature import peak_local_max
from scipy.ndimage.filters import gaussian_filter1d as smooth
from scipy.signal import find_peaks_cwt
from shapes import Circle, Rectangle, circle_to_rectangle
from keras.models import load_model
import pickle
from utils import resize_1d


def get_candidate_signals(img,
                          signal_len,
                          left_offset=5,
                          threshold_abs=50,
                          min_distance=2,
                          smooth_img=False,
                          verbose=False):

    if smooth_img:
        img = smooth(img, sigma=1, axis=0)

    local_max = peak_local_max(img,
                               min_distance=min_distance,
                               threshold_abs=threshold_abs,
                               exclude_border=True)

    signals_z = []
    signals_x = []
    signals_y = []

    for idx, lm in enumerate(local_max):
        xmin_offset = lm[1] - left_offset
        if xmin_offset < 0:
            xmin_offset = 0
        xmax_offset = lm[1] + signal_len - left_offset
        if xmax_offset > img.shape[1]:
            xmax_offset = img.shape[1] - 1

        sz = img[lm[0], xmin_offset:xmax_offset]
        sx = np.arange(xmin_offset, xmax_offset, 1)
        sy = np.ones(sz.shape) * lm[0]

        if len(sz) == len(sx):
            signals_z.append(sz)
            signals_x.append(sx)
            signals_y.append(sy)

    return signals_x, signals_y, signals_z


def get_salient_peaks(sig_z, nb_peaks=2):

    peaks_arg = find_peaks_cwt(sig_z, widths=[4])
    sorted_peaks_arg = peaks_arg[np.argsort(sig_z[peaks_arg])]
    salient_peaks_arg = sorted_peaks_arg[0:nb_peaks]

    if len(salient_peaks_arg) >= 2:
        small_peak = salient_peaks_arg[0]
        large_peak = salient_peaks_arg[1]

        return large_peak, small_peak
    else:
        print("WARNING: not enough peaks found, returning 1, np.nan")
        return 1, np.nan


def get_bubble_from_signal(sig_x, sig_y, sig_z,
                           calib_radius_func,
                           sigma=1,
                           flip_signal=False,
                           verbose=False):
    if flip_signal:
        sig_z = np.flip(sig_z, axis=0)

    sig_z = smooth(sig_z, sigma=sigma)
    first_peak_arg, second_peak_arg = get_salient_peaks(sig_z)
    radius = calib_radius_func(np.abs(second_peak_arg - first_peak_arg)/2)
    max_peak_x = sig_x[first_peak_arg]

    if verbose:
        print("first_peak_arg:", first_peak_arg)
        print("second_peak_arg", second_peak_arg)

    if np.isnan(second_peak_arg):
        return Circle(0, 0, 1)
    else:
        second_peak_x = sig_x[second_peak_arg]

        if verbose:
            print("max_peak_x", max_peak_x)
            print("second_peak_x", second_peak_x)

        cent_x = (max_peak_x + second_peak_x) / 2
        cent_y = sig_y[0]

        return Circle(cent_x, cent_y, radius)


def get_features(sig_z):
    peak1_arg, peak2_arg = get_salient_peaks(sig_z)
    if np.isnan(peak2_arg):
        peak2_z = 0
        dist_peaks = 0
        peak1_z = sig_z[peak1_arg]
    else:
        dist_peaks = peak1_arg - peak2_arg
        peak1_z = sig_z[peak1_arg]
        peak2_z = sig_z[peak2_arg]
    return [peak1_z, peak2_z, dist_peaks]


def logistic_regression_classifier(sig_z, saved_model_path="data/models/logistic_regression.pickle"):
    features = get_features(sig_z)
    with open(saved_model_path, "rb") as handle:
        model = pickle.load(handle)

    if model.predict(features) == 1:
        return True
    else:
        return False


def manual_bubble_classifier(sig_z, second_peak_min_thr=30, second_peak_max_thr=45, difference_th=20):
    first_peak_arg, second_peak_arg = get_salient_peaks(sig_z)

    if np.isnan(second_peak_arg):
        return False
    else:
        if sig_z[first_peak_arg] <= sig_z[second_peak_arg]:
            return False
        if sig_z[second_peak_arg] <= second_peak_min_thr:
            return False
        if sig_z[first_peak_arg] - sig_z[second_peak_arg] <= difference_th:
            return False
        if sig_z[second_peak_arg] >= second_peak_max_thr:
            return False

        return True


def cnn_classifier(sig_z, saved_model_path="data/models/bubbleNet1D.h5", verbose=False):
    nb_features = 20
    model = load_model(saved_model_path)
    sig_z = resize_1d(sig_z, nb_features)

    if verbose:
        print("len sig_z after", len(sig_z))
        print("loaded model")
    sig_z = np.reshape(sig_z, (1, sig_z.shape[0], 1))
    if model.predict_classes(sig_z, verbose=verbose):
        print("signal is bubble")
        return True
    else:
        print("NO bubble")
        return False


def is_bubble(sig_z, classifier_name):

    if classifier_name == "manual":
        return manual_bubble_classifier(sig_z)
    elif classifier_name == "logistic_regression":
        return logistic_regression_classifier(sig_z)
    elif classifier_name == "cnn":
        return cnn_classifier(sig_z)
    elif classifier_name == "bubble_forever":
        return True
    else:
        print("WARNING: Classifier not recognized! Using default")
        return manual_bubble_classifier(sig_z)


def detec_bubble(img,
                 calib_radius_func,
                 signal_len,
                 classifier,
                 threshold_abs=50,
                 min_distance=2,
                 output_shape="Rectangle",
                 sigma=1,
                 flip_signal=False,
                 return_signals_only=False,
                 verbose=False):
    """

    :param img:
    :param calib_radius_func:
    :param signal_len:
    :param classifier: "cnn", "logistic_regression" or "bubble_forever"
    :param threshold_abs:
    :param min_distance:
    :param output_shape: "Circle" or "Rectangle"
    :param sigma:
    :param flip_signal:
    :param return_signals_only:
    :param verbose:
    :return:
    """

    signals_x, signals_y, signals_z = get_candidate_signals(img,
                                                            signal_len,
                                                            threshold_abs=threshold_abs,
                                                            min_distance=min_distance,
                                                            smooth_img=False,
                                                            verbose=verbose)

    if return_signals_only: # debug mode
        print("# candidate signals", len(signals_x))
        print("len signal", len(signals_z[0]))
        return signals_x, signals_y, signals_z

    bubbles = [get_bubble_from_signal(sig_x, sig_y, sig_z,
                                      calib_radius_func, sigma,
                                      flip_signal, verbose=verbose)
                            for (sig_x, sig_y, sig_z) in zip(signals_x, signals_y, signals_z)
                            if is_bubble(sig_z, classifier)]

    if output_shape == "Circle":
        return bubbles
    elif output_shape == "Rectangle":
        return [circle_to_rectangle(bubb) for bubb in bubbles]
    else:
        print("WARNING: return type not supported. Returning circles")
        return bubbles


