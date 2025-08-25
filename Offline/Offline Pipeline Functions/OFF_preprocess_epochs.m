function processed_epochs = OFF_preprocess_epochs(raw_epochs, fs, apply_reference)

num_epochs = size(raw_epochs,3);
processed_epochs = zeros(size(raw_epochs));
[b,a] = butter(2, [2 20]/(fs/2), 'bandpass');

for i = 1:num_epochs

    % Extracting a single epoch i out of the raw epoch matrix
    epoch_data = squeeze(raw_epochs(:,:,i));

    % Remove DC Offset (substracting mean of each epoch = baseline removal)
    epoch_data_dc = epoch_data - mean(epoch_data, 2);

    % Bandpass Filtering (2-20Hz)
    % using butter(2, ...) + filtfilt() results in a 4th-order zero-phase filter -> filtfilt() applies the filter forward and backward.
    epoch_data_filtered = filtfilt(b, a, epoch_data_dc')';

    % Artifact Removal using MAD Thresholding with Sliding Window
    epoch_data_cleaned = remove_artifacts(epoch_data_filtered, fs, 1, 0.5, 4.5, 0.5);

    % Perform DC Offset Removal and Average Referencing after all other 
    % Signal Cleaning to avoid including artifacts into the Means.

    % Average Referencing
    if apply_reference
        epoch_data_referenced = epoch_data_cleaned - mean(epoch_data_cleaned, 1);
    else
        epoch_data_referenced = epoch_data_cleaned;   % Keep already referenced Data as it is
    end

    % Store the processed epoch i
    processed_epochs(:,:,i) = epoch_data_referenced;

    fprintf('Finished Preprocessing of Epoch %d/%d\n', i, num_epochs);

end

fprintf('Preprocessing Finished. Preprocessed Data is stored in "processed_epochs".\n');
