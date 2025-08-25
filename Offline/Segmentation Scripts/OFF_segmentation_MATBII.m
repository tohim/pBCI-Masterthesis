% Get the Data from the COG-BCI Dataset with the MATB-II Experiment Task

addpath(genpath('E:\SchuleJobAusbildung\HTW\MasterThesis\Code\EEGLAB'));
eeglab;

% Path to root folder
root_folder = 'E:\SchuleJobAusbildung\HTW\MasterThesis\Code\TrainingDatasets\Workload\COG-BCI MATB-II\Separated_OnlyMATB';

% Emotiv channel labels
selected_channels = {'AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'};

% Sampling rate and epoch settings
fs_original = 250;  % After Study internal downsampling
fs_target = 128;    % Match with STEW and other of my datasets
epoch_sec = 4;
overlap = 0.5;

epoch_length = fs_target * epoch_sec;
step_size = round((1-overlap) * epoch_length);

% Initialize Output
all_epochs = [];
all_labels = [];

% Extract eeg data

% Loop over subjects
for subj = 1:15
    subj_folder = sprintf('P%02d', subj);
    
    % Loop over sessions
    for session = 1:2
        session_folder = sprintf('S%d', session);   % S1 or S2
        eeg_folder = fullfile(root_folder, subj_folder, session_folder, 'eeg');

        if ~isfolder(eeg_folder), continue; end
       
        % Files of Interest
        condition_files = {'MATBeasy', 'MATBmed', 'MATBdiff'};

        for f = 1:length(condition_files)
            condition = condition_files{f};
            filename = dir(fullfile(eeg_folder, ['*' condition '*.set']));

            if isempty(filename), continue;end

            EEG = pop_loadset('filename', filename.name, 'filepath', eeg_folder);

            % Select Channels
            chan_indicies = find(ismember({EEG.chanlocs.labels}, selected_channels));

            if length(chan_indicies) ~= 14
                warning('Channel mismatch in %s. Skipping.', filename.name);
                continue;

            end

            eeg_data = double(EEG.data(chan_indicies,:));

            % Downsample
            eeg_data = resample(eeg_data', fs_target, EEG.srate)';

            % Label
            if strcmp(condition, 'MATBeasy')
                label = 0;
            elseif any(strcmp(condition, {'MATBdiff'}))     % In a separate run i selected "med" and "diff" to group both into label = 1 HIGH MWL
                label = 1;                                  % "any(strcmp(condition, {'MATBdiff', 'MATBmed'}))"
            else 
                continue;   % skip the resting state files
            end

            % Segment
            num_epochs = floor((size(eeg_data,2) - epoch_length) / step_size) +1;
            epochs = zeros(length(chan_indicies), epoch_length, num_epochs);
            labels = ones(num_epochs,1) * label;

            for i = 1:num_epochs
                start_idx = (i-1) * step_size + 1;
                end_idx = start_idx + epoch_length -1;
                epochs(:,:,i) = eeg_data(:,start_idx:end_idx);
            end

            all_epochs = cat(3, all_epochs, epochs);
            all_labels = [all_labels; labels];

        end
    end
end

% Shuffle to avoid order bias
rng(42);
shuff = randperm(size(all_epochs,3));
eeg_data = all_epochs(:,:,shuff);
labels = all_labels(shuff);


fprintf('Segmentation Complete!\n');
fprintf('Low workload epochs: %d\n', sum(labels==0));
fprintf('High workload epochs: %d\n', sum(labels==1));









