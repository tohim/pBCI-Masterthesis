function processed_epoch = RT_preprocess_epochs(raw_epochs,fs,b,a)

% Real-Time Preprocessing for individual EEG Epoch
% Input:
% - raw_epoch: [Channel x Time] EEG matrix for a single epoch
% - fs       : Sampling Frequency (e.g., 128 Hz)
% b, a       : Precomputed filter coefficients for Butterworth bandpass filter
% Output:
% - preprocessed_epoch: Preprocessed EEG epoch

% Bandpass Filtering (2-20Hz)
epoch_data_filtered = filtfilt(b, a, raw_epochs')';
    
% Artifact Removal using MAD Thresholding with Sliding Window    
processed_epoch = remove_artifacts(epoch_data_filtered, fs, 1, 0.5, 4.5, 0.5);

end


