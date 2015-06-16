import numpy as np
from features import mfcc, logfbank, ssc
import scipy.io.wavfile as wav
from math import ceil
from scipy import stats

'''
Extract features from audio. Features are associated to labels.
Note: You will see condition with 1 and 2. This was done in order to test rapidly different approach.
We left the configuration giving best results.
'''
def get_data(path_to_audio, path_to_labels, delimiter_char, nb_features=13):
    (rate, sig) = wav.read(path_to_audio)
    target = np.genfromtxt(path_to_labels, dtype=long, delimiter=delimiter_char)
    labels = target[:, 1]
    if 1:  # Change Window Size
        window_size = 0.1 # Default window size is milliseconds
        if 1:
            mfcc_feat = mfcc(sig, rate, window_size, 0.1, nb_features)
        if 0:
            mfcc_feat = mfcc(sig, rate, window_size, 0.1, nb_features/2)
            fbank_feat = logfbank(sig, rate, window_size, 0.1, nb_features/2)
            #ssc_feat = ssc(sig, rate, window_size, 0.1, nb_features/2)
            temp = np.empty([mfcc_feat.shape[0],nb_features]);
            for i in range(len(mfcc_feat)):
                temp1 = np.append(mfcc_feat[i], fbank_feat[i])
                np.append(temp,temp1)
            mfcc_feat = temp
    if 0: # Change Window Size and step size when aggregation is done on labels
        window_size = 0.1
        window_step = 1
        mfcc_feat = mfcc(sig, rate, window_size, window_step, nb_features)
    if 1: # Normalize features
        print "Normalizing Features"
        for col in range(nb_features):
            min_col = np.amin(mfcc_feat[:, col])
            max_col = np.amax(mfcc_feat[:, col])
            range_col = max_col - min_col
            mfcc_feat[:, col] = (mfcc_feat[:, col] - min_col) / range_col
    if 1: # Low pass features
        print "Low Pass Filtering features"
        convolute_size = 4
        count = mfcc_feat.shape[0]
        new_feat = np.empty([count, nb_features])
        for i in range(count):
            if (i < convolute_size) or (i > count - 1 - convolute_size):
                new_feat[i, :] = mfcc_feat[i, :]
            else:
                row_ = mfcc_feat[i, :]
                for row_dex in range(1, 1 + convolute_size):
                    row_ = row_ + mfcc_feat[i + row_dex, :]
                    row_ = row_ + mfcc_feat[i - row_dex, :]
                new_feat[i, :] = row_ / (convolute_size * 2 + 1)
        mfcc_feat = new_feat
    if 0: # Aggregating labels by block
        print "Aggregation of labels on", window_step,"sec"
        count = labels.shape[0]
        aggregate_size = int(window_step*10)
        size_labels = ceil(float(count)/aggregate_size)
        modified_size = min(mfcc_feat.shape[0], size_labels)
        mfcc_feat = mfcc_feat[0:modified_size,:]
        new_labels = np.empty([modified_size])
        new_index = 0
        for i in range(0,int((modified_size-1)*aggregate_size+1),aggregate_size):
            (new_labels[new_index], count_) = stats.mode(labels[i:i+aggregate_size])
            new_index += 1
        labels = new_labels
    if 0: # Low pass labels
        print "Low Pass Filtering labels"
        convolute_size = 4
        count = labels.shape[0]
        new_labels = np.empty([count])
        for i in range(count):
            if (i < convolute_size) or (i > count - 1 - convolute_size):
                new_labels[i] = labels[i]
            else:
                row_ = labels[i]
                for row_dex in range(1, 1 + convolute_size):
                    row_ = row_ + labels[i + row_dex]
                    row_ = row_ + labels[i - row_dex]
                new_labels[i] = row_ / (convolute_size * 2 + 1)
        labels = new_labels
    if 1:  # get rid of background points = class 5
        print "Removing speaking parts"
        all_sound_count = 0
        non_verbal_count = 0
        for row in labels:
            if row != 5:
                non_verbal_count += 1
            all_sound_count += 1
        new_feat = np.empty([non_verbal_count, nb_features])
        new_target = np.empty([non_verbal_count, 2])
        new_labels = np.empty([non_verbal_count])
        count = 0
        dex = 0
        for row in labels:
            if row != 5:
                new_target[count, :] = target[dex, :]
                new_feat[count, :] = mfcc_feat[dex, :]
                new_labels[count] = labels[dex]
                count += 1
            dex += 1
        mfcc_feat = new_feat
        labels = new_labels
    labels = labels.astype('int')
    n_classes = np.unique(labels)
    return [mfcc_feat, labels, n_classes]

