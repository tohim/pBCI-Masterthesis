% Get EEG files for the Heat The Chair Dataset

% data_folder = 'E:\SchuleJobAusbildung\HTW\MasterThesis\Code\TrainingDatasets\Workload\Flight_Deck_Study\data_heat_the_chair';
% files_directory = dir(fullfile(data_folder, '*.parquet'));

file_info = parquetinfo('eeg.parquet');
rf = rowfilter(["subject" "theoretical_difficulty" "phase" "EEG.AF3" "EEG.F7" "EEG.F3" "EEG.FC5" "EEG.T7" "EEG.P7" "EEG.O1" "EEG.O2" "EEG.P8" "EEG.T8" "EEG.FC6" "EEG.F4" "EEG.F8" "EEG.AF4"]);
file = parquetread('eeg.parquet', RowFilter=rf, SelectedVariableNames=["subject" "theoretical_difficulty" "phase" "EEG.AF3" "EEG.F7" "EEG.F3" "EEG.FC5" "EEG.T7" "EEG.P7" "EEG.O1" "EEG.O2" "EEG.P8" "EEG.T8" "EEG.FC6" "EEG.F4" "EEG.F8" "EEG.AF4"]);

% Initialize Variables
fs = 128;                                           % Sampling Frequency
epoch_length = 4*fs;                                % Length of 1 Epoch in Seconds
overlap = 0.5;                                      % Amount of Overlap between Epochs
step_size = round((1-overlap) * epoch_length);      % Amount of Single Step Size in Samples

% Load EEG file
eeg_file = file{:,{'subject', 'theoretical_difficulty', 'phase', 'EEG_AF3', 'EEG_F7', 'EEG_F3', 'EEG_FC5', 'EEG_T7', 'EEG_P7', 'EEG_O1', 'EEG_O2', 'EEG_P8', 'EEG_T8', 'EEG_FC6', 'EEG_F4', 'EEG_F8', 'EEG_AF4'}};   % Reads important data from the file
eeg_data = file{:,{'EEG_AF3', 'EEG_F7', 'EEG_F3', 'EEG_FC5', 'EEG_T7', 'EEG_P7', 'EEG_O1', 'EEG_O2', 'EEG_P8', 'EEG_T8', 'EEG_FC6', 'EEG_F4', 'EEG_F8', 'EEG_AF4'}};   % Reads eeg data from the file

% Ensure the data is in double precision
eeg_data = double(eeg_data);

% Initialize data to extract labels from
% Determine label based on theoretical difficulty && "phase" (task)
subjects = file{:,'subject'};
difficulty_value = file{:,'theoretical_difficulty'};
task_value = file{:,'phase'};

nbchan = size(eeg_data,1);

% % Initialize epochs matrix and epoch labels matrix
% epochs = zeros(nbchan, epoch_length, num_epochs);
% epoch_labels = zeros(num_epochs, 1) + label;

% Ensure that data is in Channel x Time format (transpose otherwise)
if size(eeg_data,1) > size(eeg_data,2)
    eeg_data = eeg_data';   % transpose
end

% First split into high and low workload - this is necessary to make sure
% to correctly only select the data that surely correlates to high and low
% workload respectively
low_workload_indicies = (difficulty_value <= 0); % & (task_value == "baseline");
high_workload_indicies = (difficulty_value >= 2) & (task_value == "flight");

eeg_low = eeg_data(:,low_workload_indicies);
eeg_high = eeg_data(:,high_workload_indicies);

% Segment high and low EEG data into 2 second Epochs with 50% overlap
num_epochs_low = floor((size(eeg_low,2) - epoch_length) / step_size) +1;
epochs_low = zeros(size(eeg_low,1), epoch_length, num_epochs_low);
labels_low = zeros(num_epochs_low,1);   % zeros for labeling low workload=0

num_epochs_high = floor((size(eeg_high,2) - epoch_length) / step_size) +1;
epochs_high = zeros(size(eeg_high,1), epoch_length, num_epochs_high);
labels_high = ones(num_epochs_high,1);  % ones for labeling high workload=1

for i=1:num_epochs_low

    % Create Start and End Index for each individual epoch_low
    start_idx = (i - 1) * step_size + 1;            % 1 129 257 etc.
    end_idx = start_idx + epoch_length -1;          % 256 384 512 etc.
    epochs_low(:,:,i) = eeg_low(:,start_idx:end_idx);

end

for i = 1:num_epochs_high

    % Create Start and End Index for each individual epoch_high
    start_idx = (i - 1) * step_size + 1;            % 1 129 257 etc.
    end_idx = start_idx + epoch_length -1;          % 256 384 512 etc.
    epochs_high(:,:,i) = eeg_high(:,start_idx:end_idx);

end

% Combine both datasets again to merge them into 1 big dataset
eeg_data = cat(3,epochs_high, epochs_low);
labels = [labels_high; labels_low]; % using ";" to concatenate labels vertically

% Shuffle the data to mix high and low workload epochs to avoid order
% biases
rng(42);
shuffeled_indicies = randperm(size(eeg_data,3));
eeg_data = eeg_data(:,:,shuffeled_indicies);
labels = labels(shuffeled_indicies);

% Debugging Check: 

% Ensure correct label assigned
disp(['Label Counts - Low Workload: ', num2str(sum(labels == 0)), ' | High Workload: ', num2str(sum(labels == 1))]);
fprintf('\n');
% Ensure correct label types
disp(unique(labels));

% Pick specific Epoch and view its Label
epoch_num = 6;
selected_epoch = eeg_data(:,:,epoch_num);
selected_label = labels(epoch_num);
disp(['Epoch ', num2str(epoch_num), ' Label ', num2str(selected_label)]);





