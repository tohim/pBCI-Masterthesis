function [features, labels, train_features, val_features, test_features, train_labels, val_labels, test_labels] = get_data(dataset_name, dataset_procs, opts)

% Get Epoch_length String
epoch_length = sprintf('%dsec', opts.epochlength);

% Determine feature tag for filename
if opts.use_features && opts.use_csp
    feature_tag = sprintf('%dwCsp', opts.num_features);          % e.g., '24wCsp'
elseif opts.use_features
    feature_tag = sprintf('%d', opts.num_features);              % e.g., '24'
elseif opts.use_csp
    feature_tag = sprintf('csp_%d', opts.num_csp_filters);       % Generally should be 4 CSP components (2 per class)
else
    error('Unknown feature configuration: both use_features and use_csp are false.');
end

% Include sample count tag (optional if > 0)
if isfield(opts, 'total_samples') && opts.total_samples > 0
    sample_tag = sprintf('%d', opts.total_samples);
    label_base  = sprintf('%s_%s_%s', sample_tag, epoch_length, dataset_name);
    feat_base  = sprintf('%s_%s_%s_%s_%s',sample_tag, feature_tag, epoch_length, dataset_procs, dataset_name);
else
    label_base  = sprintf('%s_%s', epoch_length, dataset_name);
    feat_base  = sprintf('%s_%s_%s_%s', feature_tag, epoch_length, dataset_procs, dataset_name);
end

% Load features
train_feat_file = sprintf('%s_train_features.mat', feat_base);
val_feat_file   = sprintf('%s_val_features.mat', feat_base);
test_feat_file  = sprintf('%s_test_features.mat', feat_base);

loaded_features_train = load(train_feat_file);
loaded_features_val   = load(val_feat_file);
loaded_features_test  = load(test_feat_file);

train_features = loaded_features_train.train_features;
val_features   = loaded_features_val.val_features;
test_features  = loaded_features_test.test_features;
features       = [train_features; val_features; test_features];

% Load labels
labels_file     = sprintf('%s_sampled_labels.mat', label_base);
train_labels_file = sprintf('%s_train_labels.mat', label_base);
val_labels_file   = sprintf('%s_val_labels.mat', label_base);
test_labels_file  = sprintf('%s_test_labels.mat', label_base);

loaded_labels      = load(labels_file);
train_labels_load  = load(train_labels_file);
val_labels_load    = load(val_labels_file);
test_labels_load   = load(test_labels_file);

% Handle possible field name differences
if isfield(loaded_labels, 'downsampled_labels')
    labels = loaded_labels.downsampled_labels;
elseif isfield(loaded_labels, sprintf('downsampled_%s_labels_v2', dataset_name))
    labels = loaded_labels.(sprintf('downsampled_%s_labels_v2', dataset_name));     % Previous Naming, just to catch and get a warning in case its used
    warning('Old Version used: %s', labels_file);
else
    error('Could not find valid label field in %s', labels_file);
end

train_labels = train_labels_load.train_labels;
val_labels   = val_labels_load.val_labels;
test_labels  = test_labels_load.test_labels;

% Sanity check
if size(features, 1) ~= size(labels, 1)
    error('Mismatch between number of feature rows and label rows!');
end

fprintf('[%s] Dataset loaded successfully using features: %s\n', dataset_name, feature_tag);

end
