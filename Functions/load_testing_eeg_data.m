function [eeg_data, labels] = load_testing_eeg_data(file_list, fs, epoch_sec, overlap)
% Load and segment selected STEW EEG files into epochs
%
% Inputs:
%   - file_list: cell array of filenames (full paths)
%   - fs: sampling frequency (e.g., 128)
%   - epoch_sec: length of each epoch in seconds
%   - overlap: proportion of overlap (e.g., 0.5)
%
% Outputs:
%   - eeg_data: [channels x samples x epochs]
%   - labels: [epochs x 1]

    epoch_length = fs * epoch_sec;
    step_size = round((1 - overlap) * epoch_length);

    all_epochs = [];
    all_labels = [];

    for i = 1:length(file_list)
        file_path = file_list{i};
        eeg = readmatrix(file_path);

        % Transpose if needed
        if size(eeg,1) > size(eeg,2)
            eeg = eeg';
        end

        if contains(lower(file_path), 'hi')
            label = 1;
        elseif contains(lower(file_path), 'lo')
            label = 0;
        else
            warning('Skipping unrecognized file: %s', file_path);
            continue;
        end

        nbchan = size(eeg,1);
        num_epochs = floor((size(eeg,2) - epoch_length) / step_size) + 1;

        epochs = zeros(nbchan, epoch_length, num_epochs);
        epoch_labels = repmat(label, num_epochs, 1);

        for j = 1:num_epochs
            idx1 = (j-1) * step_size + 1;
            idx2 = idx1 + epoch_length - 1;
            epochs(:,:,j) = eeg(:, idx1:idx2);
        end

        all_epochs = cat(3, all_epochs, epochs);
        all_labels = [all_labels; epoch_labels];
    end

    % Shuffle
    rng(42);  % for reproducibility
    order = randperm(size(all_epochs,3));
    eeg_data = all_epochs(:,:,order);
    labels = all_labels(order);

end
