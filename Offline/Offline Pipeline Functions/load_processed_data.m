function [eeg_data_processed, labels] = load_processed_data(opts)

proc_filename = sprintf('%dsec_%s_%s_epochs.mat', opts.epochlength, opts.proc, opts.dataset);
label_filename = sprintf('%dsec_%s_labels.mat', opts.epochlength, opts.dataset);

fprintf('Loaded Processed Data: %s ' , proc_filename);
fprintf('\nLoaded Labels:         %s', label_filename);

loaded_data = load(proc_filename);
eeg_data_processed = loaded_data.eeg_data_processed;

loaded_labels = load(label_filename);
labels = loaded_labels.labels;

end