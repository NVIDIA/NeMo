#! /usr/bin/python

# Voice Activity Detection (VAD) tool.
# use the vad_help() function for instructions.
# Navid Shokouhi December 2012.

# Updated: May 2017 for Speaker Recognition collaboration.

import librosa

# from audio_tools import *
import numpy as np
from scipy.io import wavfile


def add_wgn(s, var=1e-4):
    """
        Add white Gaussian noise to signal
        If no variance is given, simply add jitter. 
        Jitter helps eliminate all-zero values.
        """
    np.random.seed(0)
    noise = np.random.normal(0, var, len(s))
    return s + noise


def read_wav(filename, offset, duration, sample_rate=16000):
    """
        read wav file.
        Normalizes signal to values between -1 and 1.
        Also add some jitter to remove all-zero segments."""
    #     fs, s = wavfile.read(filename)  # scipy reads int
    audio, sample_rate = librosa.load(filename, sr=sample_rate, offset=offset, duration=duration)
    audio = np.array(audio) / float(max(abs(audio)))
    audio = add_wgn(audio)  # Add jitter for numerical stability
    return audio, sample_rate


# ===============================================================================
def enframe(x, win_len, hop_len):
    """
        receives a 1D numpy array and divides it into frames.
        outputs a numpy matrix with the frames on the rows.
        """
    x = np.squeeze(x)
    if x.ndim != 1:
        raise TypeError("enframe input must be a 1-dimensional array.")
    n_frames = 1 + np.int(np.floor((len(x) - win_len) / float(hop_len)))
    x_framed = np.zeros((n_frames, win_len))
    for i in range(n_frames):
        x_framed[i] = x[i * hop_len : i * hop_len + win_len]
    return x_framed


def deframe(x_framed, win_len, hop_len):
    '''
        interpolates 1D data with framed alignments into persample values.
        This function helps as a visual aid and can also be used to change 
        frame-rate for features, e.g. energy, zero-crossing, etc.
        '''

    n_frames = len(x_framed)
    n_samples = n_frames * hop_len + win_len
    x_samples = np.zeros((n_samples, 1))
    for i in range(n_frames):
        x_samples[i * hop_len : i * hop_len + win_len] = x_framed[i]
    return x_samples


#### Display tools
def plot_this(s, title=''):
    """
     
    """
    import pylab

    s = s.squeeze()
    if s.ndim == 1:
        pylab.plot(s)
    else:
        pylab.imshow(s, aspect='auto')
        pylab.title(title)
    pylab.show()


def plot_these(s1, s2):
    import pylab

    try:
        # If values are numpy arrays
        pylab.plot(s1 / max(abs(s1)), color='red')
        pylab.plot(s2 / max(abs(s2)), color='blue')
    except:
        # Values are lists
        pylab.plot(s1, color='red')
        pylab.plot(s2, color='blue')
    #     pylab.legend()
    pylab.show()


#### Energy tools
def zero_mean(xframes):
    """
        remove mean of framed signal
        return zero-mean frames.
        """
    m = np.mean(xframes, axis=1)
    xframes = xframes - np.tile(m, (xframes.shape[1], 1)).T
    return xframes


def compute_nrg(xframes):
    # calculate per frame energy
    n_frames = xframes.shape[1]
    return np.diagonal(np.dot(xframes, xframes.T)) / float(n_frames)


def compute_log_nrg(xframes, norm=None):
    # calculate per frame energy in log
    n_frames = xframes.shape[1]
    raw_nrgs = np.log(compute_nrg(xframes + 1e-5)) / float(n_frames)

    """
    The log-energy values are subject to mean and var
        normalization to simplify the picking the right threshold. 
        In this framework, the default threshold is 0.0
        
    """
    if norm == "mvn":
        return (raw_nrgs - np.mean(raw_nrgs)) / (np.sqrt(np.var(raw_nrgs))), 0.0

    minimum = np.min(raw_nrgs)
    demo = np.max(raw_nrgs) - minimum

    norms = (raw_nrgs - minimum) / demo
    thr = np.mean(norms)
    return norms, thr


def power_spectrum(xframes):
    """
        x: input signal, each row is one frame
        """
    X = np.fft.fft(xframes, axis=1)
    X = np.abs(X[:, : X.shape[1] / 2]) ** 2
    return np.sqrt(X)


def nrg_vad(xframes, percent_thr, nrg_thr=None, context=5):
    """
        Picks frames with high energy as determined by a 
        user defined threshold.
        
        This function also uses a 'context' parameter to
        resolve the fluctuative nature of thresholding. 
        context is an integer value determining the number
        of neighboring frames that should be used to decide
        if a frame is voiced.
        
        The log-energy values are subject to mean and var
        normalization to simplify the picking the right threshold. 
        In this framework, the default threshold is 0.0
        """

    xframes = zero_mean(xframes)
    n_frames = xframes.shape[0]

    # Compute per frame energies:
    xnrgs, thr = compute_log_nrg(xframes)

    if not nrg_thr:
        nrg_thr = thr
    #     print("nrg_thr", nrg_thr)
    xvad = np.zeros((n_frames, 1))
    for i in range(n_frames):
        start = max(i - context, 0)
        end = min(i + context, n_frames - 1)
        n_above_thr = np.sum(xnrgs[start:end] > nrg_thr)
        n_total = end - start + 1
        xvad[i] = 1.0 * ((float(n_above_thr) / n_total) > percent_thr)

    return xnrgs, xvad
