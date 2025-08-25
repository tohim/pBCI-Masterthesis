% Get EEG files for the STEW Dataset

data_folder = 'E:\SchuleJobAusbildung\HTW\MasterThesis\Code\TrainingDatasets\Workload\STEW Dataset';
files = dir(fullfile(data_folder, '*.txt'));

% Initialize storage for EEG data and labels
all_epochs = [];
all_labels = [];

% Initialize Variables
fs = 128;                                           % Sampling Frequency
epoch_length = 4*fs;                                % Length of 1 Epoch in Seconds
overlap = 0.5;                                      % Amount of Overlap between Epochs
step_size = round((1-overlap) * epoch_length);      % Amount of Single Step Size in Samples

for file = files'
    % Load EEG file
    eeg_data = readmatrix(file.name);   % Reads numerical data from the file

    % Ensure that data is in Channel x Time format (transpose otherwise)
    if size(eeg_data,1) > size(eeg_data,2)
        eeg_data = eeg_data';   % transpose
    end

    % Determine label based on filenime
    if contains(file.name, 'hi')
        label = 1;  % High Workload
    elseif contains(file.name, 'lo')
        label = 0;  % Low Workload
    else
        continue;   % Skip if filename doesnt match expected pattern
    end

    nbchan = size(eeg_data,1);

    % Segment EEG data into x second Epochs with 50% overlap
    num_epochs = floor((size(eeg_data,2) - epoch_length) / step_size) +1;

    % Initialize epochs matrix and epoch labels matrix
    epochs = zeros(nbchan, epoch_length, num_epochs);
    epoch_labels = zeros(num_epochs, 1) + label;

    for i = 1:num_epochs
        start_idx = (i - 1) * step_size + 1;            % 1 129 257 etc.
        end_idx = start_idx + epoch_length -1;          % 256 384 512 etc.
        epochs(:,:,i) = eeg_data(:,start_idx:end_idx);
    end

    all_epochs = cat(3, all_epochs, epochs);
    all_labels = [all_labels; epoch_labels];    % Add all epoch labels (all labels for each epoch of 1 file)

end

% Store Final Datasets
eeg_data = all_epochs;
labels = all_labels;

% Shuffle epochs to avoid ordering bias
rng(42);
shuffled_indices = randperm(size(eeg_data, 3));
eeg_data = eeg_data(:,:,shuffled_indices);
labels = labels(shuffled_indices);


% Debugging Check: 
% Ensure correct label assigned
disp(['Label Counts:  Low Workload: ', num2str(sum(labels == 0)), ' | High Workload: ', num2str(sum(labels == 1))]);

fprintf('\n');

% Ensure correct label types
disp(unique(labels));

% Pick specific Epoch and view its Label
epoch_num = 6;
selected_epoch = eeg_data(:,:,epoch_num);
selected_label = labels(epoch_num);
disp(['Epoch ', num2str(epoch_num), ' Label ', num2str(selected_label)]);


%% Because STEW dataset is already preprocessed with referencing, filter and ASR 
% it is saved instantly as "eeg_data_preprocessed"

%eeg_data_preprocessed = eeg_data;

%fprintf('Because STEW dataset is already preprocessed with referencing, filter and ASR it is saved instantly as "eeg_data_preprocessed"');


