#!/usr/bin/python3
# coding: utf-8

import sys
import os
sys.path.insert(0,'/home/ced/Coding/EEGLearn/')



from braindecode.datautil.splitters import split_into_train_valid_test, split_into_train_test
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from collections import OrderedDict




def load_file(filepath, low_cut_hz, high_cut_hz):
    label_filepath = filepath.replace('.gdf', '.mat')
    loader = BCICompetition4Set2A(
        filepath, labels_filename=label_filepath)
    cnt = loader.load()

    # Preprocessing
    cnt = cnt.drop_channels([
        'STI 014', 'EOG-left', 'EOG-central', 'EOG-right'])
    assert len(cnt.ch_names) == 22
    # lets convert to millvolt for numerical stability of next operations
    cnt = mne_apply(lambda a: a * 1e6, cnt)
    cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), cnt)
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        cnt)

    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])
    ival = [-500, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    return dataset

def load_data(data_folder, subject_id, low_cut_hz, high_cut_hz, train_or_test):
    if train_or_test == 'train':
        filename = 'A{:02d}T.gdf'.format(subject_id)
    else:
        assert train_or_test == 'test'
        filename = 'A{:02d}E.gdf'.format(subject_id)
    filepath = os.path.join(data_folder, filename)
    dataset = load_file(filepath, low_cut_hz, high_cut_hz)
    return dataset


n_folds = 5
i_test_fold = 4
data_folder = '~/Coding/EEGLearn/data_gdf/'
subject_id = 1 # from 1-9
low_cut_hz = 7 # or 4, as before
high_cut_hz = 14
train_set = load_data(data_folder, subject_id, low_cut_hz, high_cut_hz, "train")

# train_set, valid_set, test_set = split_into_train_valid_test(train_set, n_folds=n_folds, i_test_fold=i_test_fold,)

# train_set, test_set = split_into_train_test(train_set, n_folds=n_folds, i_test_fold=i_test_fold,)

