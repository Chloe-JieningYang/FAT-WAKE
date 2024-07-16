import numpy as np
import scipy
from scipy import signal
import pandas as pd
import entropy_feature as enft

# notch filter of 50Hz (cuz of the AC power)
def Notch_Filter(sig_data, sample_rate):
    f0 = 50
    fs = sample_rate
    Q = 20
    b, a = signal.iirnotch(f0, Q, fs)
    filted_data = signal.filtfilt(b, a, sig_data)
    return filted_data

# band filter of 0.15Hz-45Hz
def BandFilter(sig_data, sample_rate, lowcut=0.15, highcut=45, order=8):
    nyquist_freq = 0.5 * sample_rate
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq

    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.lfilter(b, a, sig_data)

    return filtered_data

# FFT with a hamming window, returning the frequency data and number of groups
# return the Fourier transform values and the frequency of the values
def FFT_ham(sig_data, sample_rate):
    N = 5 * sample_rate
    hamming_win = signal.hamming(N)
    sig_win = sig_data * hamming_win
    sig_fft = np.fft.fft(sig_win)
    f = np.fft.fftfreq(N, 1 / sample_rate)
    return sig_fft, f

# mean normalization
def mean_normalization(x):
    y = (x - x.mean()) / (x.max() - x.min())
    return y

# get features (maybe will add other features later)
def get_features(file_list, x1, x2, sample_rate):
    print("get features start")
    AE = []
    SE = []
    FE = []
    PE = []
    delta = []
    theta = []
    alpha = []
    beta = []
    gamma = []
    fi = []
    nyquist_freq = 0.5 * sample_rate
    for file in file_list:
        print(f"{file} begin")
        eeg = np.loadtxt(file)

        # cut the full data into 5s/piece
        for i in range(x1, x2):
            print(f"{i}s begin")
            eeg_data = eeg[i * sample_rate * 5:(i + 1) * sample_rate * 5]
            # print(eeg_data)
            eeg_data = mean_normalization(eeg_data)
            notch_filtered_data = Notch_Filter(eeg_data, sample_rate)
            filtered_data = BandFilter(notch_filtered_data, sample_rate)

            # get entropy feature
            AE.append(enft.AE(filtered_data))
            SE.append(enft.SE(filtered_data))
            FE.append(enft.FE(filtered_data))
            PE.append(enft.PE(filtered_data))

            # get frequency feature
            sig_fft, freq_fft = FFT_ham(filtered_data, sample_rate)
            sig_power = np.abs(sig_fft) ** 2
            # print(f"freq_fft:{freq_fft},max={freq_fft.max()}")
            useful_sig_power = sig_power[:int(nyquist_freq * 5)]
            delta.append(np.mean(useful_sig_power[int(0.5 * 5):int(4 * 5)]))
            theta.append(np.mean(useful_sig_power[int(4 * 5):int(8 * 5)]))
            alpha.append(np.mean(useful_sig_power[int(8 * 5):int(13 * 5)]))
            beta.append(np.mean(useful_sig_power[int(13 * 5):int(30 * 5)]))
            gamma.append(np.mean(useful_sig_power[int(30 * 5):int(48 * 5)]))
            fi.append(np.mean(useful_sig_power[int(0.85 * 5):int(110 * 5)]))

    print("get features end")
    return np.array(AE), np.array(SE), np.array(FE), np.array(PE), np.array(delta), np.array(theta), np.array(
        alpha), np.array(beta), np.array(gamma), np.array(fi)

# get KSS
# zero_one means if it is a binary classification(True)
def get_KSS(file_name, x1, x2, zero_one=True):
    print("get kss start")
    data = pd.read_excel(file_name, header=0)

    wake_kss = np.array(data[['wake_kss']])
    fat_kss = np.array(data[['fat_kss']])

    if zero_one:
        wake_kss = np.zeros(wake_kss.shape)
        fat_kss = np.ones(fat_kss.shape)

    repeated_wake_kss = np.repeat(wake_kss, x2 - x1)
    repeated_fat_kss = np.repeat(fat_kss, x2 - x1)

    print("get kss end")
    # print(f"shape of KSS:{repeated_fat_kss.shape, repeated_wake_kss.shape}")
    return repeated_wake_kss, repeated_fat_kss
