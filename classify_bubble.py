import numpy as np
from skimage.feature import peak_local_max
from scipy.ndimage.filters import gaussian_filter1d as smooth
from scipy.signal import find_peaks_cwt
from shapes import Circle, Rectangle, circle_to_rectangle
from keras.models import load_model
import pickle


def get_candidate_signals(img,
                          signal_len,
                          threshold_abs=50,
                          min_distance=2,
                          smooth_img=False):

    offset = signal_len

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
        sz = img[lm[0], lm[1]-offset:lm[1]+offset]
        sx = np.arange(lm[1]-offset, lm[1]+offset, 1)

        if len(sz) == len(sx):
            signals_z.append(sz)
            signals_x.append(sx)
            signals_y.append(lm[0])

    return signals_x, signals_y, signals_z


def get_salient_peaks(sig_z):

    peaks_arg = find_peaks_cwt(sig_z, widths=np.arange(3, 4))
    first_peak_arg = sig_z[peaks_arg].argmax()

    if first_peak_arg >= 1:
        second_peak_arg = first_peak_arg - 1
        return peaks_arg[first_peak_arg], peaks_arg[second_peak_arg]
    else:
        return peaks_arg[first_peak_arg], np.nan


def get_bubble_from_signal(sig_x, sig_y, sig_z, calib_radius_func, signal_inverted):
    if signal_inverted:
        sig_z = np.invert(sig_z)

    first_peak_arg, second_peak_arg = get_salient_peaks(sig_z)
    radius = calib_radius_func(np.abs(second_peak_arg - first_peak_arg))
    max_peak_x = sig_x[first_peak_arg]

    if np.isnan(second_peak_arg):
        return Circle(0, 0, 0)
    else:
        second_peak_x = sig_x[second_peak_arg]

        cent_x = int(((max_peak_x - second_peak_x) / 2) + second_peak_x)
        cent_y = int(sig_y)

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


def cnn_classifier(sig_z, saved_model_path="data/models/bubbleNet1D.h5"):
    model = load_model(saved_model_path)
    sig_z = np.reshape(sig_z, (1, sig_z.shape[0], 1))
    if model.predict_classes(sig_z, verbose=0):
        return True
    else:
        return False


def is_bubble(sig_z, classifier_name):

    if classifier_name == "manual":
        return manual_bubble_classifier(sig_z)
    elif classifier_name == "logistic_regression":
        return logistic_regression_classifier(sig_z)
    elif classifier_name == "cnn":
        return cnn_classifier(sig_z)
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
                 signal_inverted=False):

    signals_x, signals_y, signals_z = get_candidate_signals(img,
                                                            signal_len,
                                                            threshold_abs=threshold_abs,
                                                            min_distance=min_distance,
                                                            smooth_img=False)

    bubbles = [get_bubble_from_signal(sig_x, sig_y, sig_z, calib_radius_func, signal_inverted)
                            for (sig_x, sig_y, sig_z) in zip(signals_x, signals_y, signals_z)
                            if is_bubble(sig_z, classifier)]

    if output_shape == "Circle":
        return bubbles
    elif output_shape == "Rectangle":
        return [circle_to_rectangle(bubb) for bubb in bubbles]
    else:
        print("WARNING: return type not supported. Returning circles")
        return bubbles


