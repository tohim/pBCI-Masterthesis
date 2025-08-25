epoch = 98;

eeg_data_raw = calibration_log(epoch).raw;  % Get the raw EEG [14 x 512]
eeg_data_processed = calibration_log(epoch).processed;  % Get the processed EEG [14 x 512]

figure;
subplot(2,1,1)
plot(eeg_data_raw');  % Transpose to plot channels over time (512 samples on x-axis)
xlabel('Sample Index');
ylabel('Amplitude (µV)');
title('Raw EEG');
legend(arrayfun(@(ch) sprintf('Ch %d', ch), 1:size(eeg_data_raw,1), 'UniformOutput', false));
grid on;

subplot(2,1,2)
plot(eeg_data_processed');  % Transpose to plot channels over time (512 samples on x-axis)
xlabel('Sample Index');
ylabel('Amplitude (µV)');
title('Processed EEG');
legend(arrayfun(@(ch) sprintf('Ch %d', ch), 1:size(eeg_data_processed,1), 'UniformOutput', false));
grid on;

% eeg_data = calibration_log(1).processed;  % Get the first entry's raw EEG [14 x 512]
% 
% figure;
% plot(eeg_data');  % Transpose to plot channels over time (512 samples on x-axis)
% xlabel('Sample Index');
% ylabel('Amplitude (µV)');
% title('Processed EEG - Block 1 Sample 1');
% legend(arrayfun(@(ch) sprintf('Ch %d', ch), 1:size(eeg_data,1), 'UniformOutput', false));
% grid on;