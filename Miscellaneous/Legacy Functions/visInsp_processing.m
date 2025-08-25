%% Visual Inspection before vs after preprocessing

% Load Raw and Processed Datasets for Comparison
loaded_raw_data = load('4sec_raw_HEATCHAIR_epochs.mat');
eeg_data = loaded_raw_data.eeg_data;

loaded_processed_data = load('4sec_processed_HEATCHAIR_ref_2to40Hz_butter2ndorder_ASR.mat');
eeg_data_processed = loaded_processed_data.processed_eeg;

% Detrend raw data to see if it removes 1 hz component
detrended_raw = zeros(size(eeg_data));

for i = 1:size(eeg_data,3)
    detrended_raw(:,:,i) = detrend(eeg_data(:,:,i));    % apply detrending to each epoch
end

% Test for Amplitude differences between Raw and Preprocessed

% First reference the raw epoch to make it more comparable to the
% preprocessed signal
raw_ref = zeros(size(eeg_data,1), size(eeg_data,3));    % Each channel gets its own reference per Epoch
raw_referenced = zeros(size(eeg_data));

for i = 1:size(eeg_data,3)
    raw_ref(:,i) = mean(eeg_data(:,:,i), 2);            % average across timepoints
    raw_referenced(:,:,i) = eeg_data(:,:,i) - raw_ref(:,i);
end

% Check Max values of raw, raw referenced and processed data
% max_raw = max(eeg_data(:));
% max_raw_referenced = max(raw_referenced(:));
% max_processed = max(eeg_data_processed(:));
% disp(['Max Raw: ', num2str(max_raw), ' | Max Referenced: ', num2str(max_raw_referenced),' | Max Processed: ', num2str(max_processed)]);

% Plot single epoch before, during and after preprocessing
sample_epoch = 300;

% get some raw and processed data sample
% take 1 sample epoch from the segmented eeg_data
sample_epoch_raw = eeg_data(:,:,sample_epoch);                                         
sample_epoch_detrend = detrended_raw(:,:,sample_epoch);                         
sample_epoch_raw_referenced = raw_referenced(:,:,sample_epoch);

% get processed version of the same epoch
sample_epoch_processed = eeg_data_processed(:,:,sample_epoch);

% Normalize epochs (z-score normalization)
sample_epoch_raw = (sample_epoch_raw - mean(sample_epoch_raw,2)) ./ (std(sample_epoch_raw,0,2)+eps);
sample_epoch_detrend = (sample_epoch_detrend - mean(sample_epoch_detrend,2)) ./ (std(sample_epoch_detrend,0,2)+eps);
sample_epoch_raw_referenced = (sample_epoch_raw_referenced - mean(sample_epoch_raw_referenced,2)) ./ (std(sample_epoch_raw_referenced,0,2)+eps);
sample_epoch_processed = (sample_epoch_processed - mean(sample_epoch_processed,2)) ./ (std(sample_epoch_processed,0,2)+eps);

% Epoch Time Vector
epoch_time = linspace(0,epoch_length/fs,epoch_length);

% Plot both Samples to see difference
figure();
hold on;
plot(epoch_time, mean(sample_epoch_raw,1), 'r', 'LineWidth',1.5)
plot(epoch_time, mean(sample_epoch_raw_referenced,1), 'b', 'LineWidth',1.5)
plot(epoch_time, mean(sample_epoch_detrend,1), 'm', 'LineWidth',1.5)
plot(epoch_time, mean(sample_epoch_processed,1), 'g', 'LineWidth',1.5);
legend('Raw', 'Raw Referenced', 'Raw EEG Detrended', 'Processed EEG');
xlabel('Time (s)')
ylabel('Amplitude (Normalized)')
title('Comparison of EEG epoch samples Before vs After Preprocessing')
grid on;

% Check the power spectrums before and after referencing and after
% processing
psd_raw = abs(fft(mean(eeg_data(:,:,sample_epoch),1))).^2;          % mean over time (averaging all channels together)
psd_detrend = abs(fft(mean(detrended_raw(:,:,sample_epoch),1))).^2; 
psd_ref = abs(fft(mean(sample_epoch_raw_referenced,1))).^2;             
psd_proc = abs(fft(mean(sample_epoch_processed,1))).^2;             

% Normalize by its total power
psd_raw_norm = psd_raw/sum(psd_raw);
psd_de_norm = psd_detrend/sum(psd_detrend);
psd_proc_norm = psd_proc/sum(psd_proc);
psd_ref_norm = psd_ref/sum(psd_ref);

% Frequency Axis 
N = length(psd_raw);
freqs = (0:N/2-1) * (fs/N);

figure();
hold on;
plot(freqs,psd_raw_norm(1:N/2),'r');
plot(freqs,psd_ref(1:N/2), 'b');
plot(freqs,psd_de_norm(1:N/2), 'm');
plot(freqs,psd_proc_norm(1:N/2), 'g');
legend('Raw EEG', 'EEG Ref', 'EEG Detrend', 'Processed EEG');
title('Power Spectral Density Comparison')
