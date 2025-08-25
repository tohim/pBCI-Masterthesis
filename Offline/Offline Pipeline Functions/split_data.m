function [train_epochs, val_epochs, test_epochs, train_labels, val_labels, test_labels] = split_data(eeg_data, labels, opts)
    
    % Find idx with according labels
    idx_low = find(labels == 0);
    idx_high = find(labels == 1);

    % Shuffle within Class
    rng(42);
    idx_low = idx_low(randperm(length(idx_low)));
    idx_high = idx_high(randperm(length(idx_high)));
    
    % Get total per class (takes smaller one to balance properly)
    min_class = min(length(idx_low), length(idx_high));

    % If opts.total_samples --> Limit the number of samples accordingly
    if isfield(opts, 'total_samples') && opts.total_samples > 0

        % Reduce each class to half of total samples (balanced split)
        max_per_class = floor(opts.total_samples / 2);

        if max_per_class > min_class
            warning('\nRequested more samples (%d per class) than available. Using max available: %d per class.', max_per_class, min_class);
            max_per_class = min_class;
        end
        
        % Trim both to same size
        idx_low = idx_low(1:max_per_class);
        idx_high = idx_high(1:max_per_class);
        fprintf('\n[INFO] Using %d samples total (%d per class).\n', 2 * max_per_class, max_per_class);
    else
        % Use max available balanced subset
        idx_low = idx_low(1:min_class);
        idx_high = idx_high(1:min_class);
        fprintf('\n[INFO] Using all available balanced samples: %d total (%d per class).\n', 2 * min_class, min_class);
    end

    % Split each class into 70/15/15
    num_train = round(0.7 * length(idx_low));
    num_val   = round(0.15 * length(idx_low));

    train_idx = [idx_low(1:num_train); idx_high(1:num_train)];
    val_idx   = [idx_low(num_train+1:num_train+num_val); idx_high(num_train+1:num_train+num_val)];
    test_idx  = [idx_low(num_train+num_val+1:end); idx_high(num_train+num_val+1:end)];

    train_idx = train_idx(randperm(length(train_idx)));
    val_idx   = val_idx(randperm(length(val_idx)));
    test_idx  = test_idx(randperm(length(test_idx)));

    train_epochs = eeg_data(:,:,train_idx);
    val_epochs   = eeg_data(:,:,val_idx);
    test_epochs  = eeg_data(:,:,test_idx);
    train_labels = labels(train_idx);
    val_labels   = labels(val_idx);
    test_labels  = labels(test_idx);

    downsampled_labels = [train_labels; val_labels; test_labels];

    % Optional: save splits
    tag = sprintf('%d_%dsec_%s', opts.total_samples, opts.epochlength, opts.dataset);

    save(sprintf('%s_train_labels.mat', tag), 'train_labels');
    save(sprintf('%s_val_labels.mat', tag), 'val_labels');
    save(sprintf('%s_test_labels.mat', tag), 'test_labels');
    save(sprintf('%s_sampled_labels.mat', tag), 'downsampled_labels');

    % Save the corresponding indicies in case i need to test or reproduce
    % the results
    save(sprintf('%s_split_indices.mat', tag), 'train_idx', 'val_idx', 'test_idx');
end