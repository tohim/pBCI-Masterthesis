function [acc1, acc2, calib_info, params, per_class_table_source, per_class_table_cross] = run_calibration(params)

% close all;
% -------------------------------------------------------------------------
% CROSS-DATA CALIBRATION
% -------------------------------------------------------------------------
fprintf('\n\n[INFO] Starting Cross-Data Model Calibration and Evaluation... \n');


% -------------------------------------------------------------------------
% Get Current Feature Configuration
% -------------------------------------------------------------------------
if params.use_features && params.use_csp
    params.feature_names = params.combined_feature_names;
    feature_tag = sprintf('%dwCsp', params.num_features);           % e.g., '24wCsp'
elseif params.use_features
    params.feature_names = params.handcrafted_feature_names;
    feature_tag = sprintf('%d', params.num_features);               % e.g., '24'
elseif params.use_csp
    params.feature_names = params.csp_feature_names;
    feature_tag = sprintf('csp_%d', params.num_csp_filters);        % e.g., 'csp_4'
else
    error('Unknown feature configuration (neither handcrafted nor CSP)');
end


% -------------------------------------------------------------------------
% Inference of Calibration Type and Model Type
% -------------------------------------------------------------------------
%params.calibration = 'adapted';              % Calibration Type                  = 'adapted', 'finetuned', 'finetuned_adapted'
%params.modeltype = 'norm_model';             % Standard or Normalized Model Type = 'model', 'norm_model'

if params.only_domain_adaptation
    params.calibration = 'adapted';
    params.modeltype = 'norm_model';
elseif params.do_domain_adaptation && params.do_transfer_learning
    params.calibration = 'finetuned_adapted';
    params.modeltype = 'norm_model';
elseif params.do_transfer_learning && ~params.do_domain_adaptation
    params.calibration = 'finetuned';
    params.modeltype = 'model';
end

% -------------------------------------------------------------------------
% Calibration Data Selection
% -------------------------------------------------------------------------
% Selecting data from the train and cross-dataset to test for calibration

% Get Train Data
[~, ~, train_features, val_features, test_features, ~, val_labels, test_labels] = get_data(params.dataset, params.proc, params);

% Get Calibration Data
[cross_features, cross_labels] = get_data(params.calibrationset, params.cross_proc, params);

% Select Calibration Subset (as per RT Pipeline Calibration Phase)
% Specify Calibration Samples Needed:
samples_per_class = params.calib_samples / 2; % Half of Calibration Samples (for each class one half)
train_ratio = 0.7;
total_required_samples = round(params.calib_samples / train_ratio);
samples_per_class_total = round(total_required_samples / 2);
val_test_samples_per_class = round((samples_per_class_total - samples_per_class)/2);

% Take small Sample Size of the Cross-Data to use for fine-tuning
low_idx = find(cross_labels == 0);
high_idx = find(cross_labels == 1);
rng(42);
low_idx = low_idx(randperm(length(low_idx)));
high_idx = high_idx(randperm(length(high_idx)));

% Split 70% / 15% / 15%
train_idx = [low_idx(1:samples_per_class); high_idx(1:samples_per_class)];
val_idx = [low_idx(samples_per_class+1:samples_per_class+val_test_samples_per_class); high_idx(samples_per_class+1:samples_per_class+val_test_samples_per_class)];
test_idx = [low_idx(samples_per_class+val_test_samples_per_class+1:samples_per_class_total); high_idx(samples_per_class+val_test_samples_per_class+1:samples_per_class_total)];
train_idx = train_idx(randperm(length(train_idx)));
val_idx = val_idx(randperm(length(val_idx)));
test_idx = test_idx(randperm(length(test_idx)));

% Training Data
tuning_train_features = cross_features(train_idx,:);
tuning_train_labels = cross_labels(train_idx);

% Validation and Testing Data (remaining unseen data)
tuning_val_features = cross_features(val_idx,:);
tuning_val_labels = cross_labels(val_idx);

% For now not using the test data - will be used if further 
% Hyperparameter Tuning included
tuning_test_features = cross_features(test_idx,:);
tuning_test_labels = cross_labels(test_idx);

% Display structure
fprintf('\n=== Final Stratified Split (Fully Balanced) ===\n');
fprintf('Train: %d samples |  Low: %d | High: %d\n', ...
    length(tuning_train_labels), sum(tuning_train_labels==0), sum(tuning_train_labels==1));
fprintf('Val:   %d samples |  Low: %d | High: %d\n', ...
    length(tuning_val_labels), sum(tuning_val_labels==0), sum(tuning_val_labels==1));
fprintf('Test:  %d samples |  Low: %d | High: %d\n', ...
    length(tuning_test_labels), sum(tuning_test_labels==0), sum(tuning_test_labels==1));


% -------------------------------------------------------------------------
% Build Model Filename
% -------------------------------------------------------------------------
epoch_length = sprintf('%dsec', params.epochlength);
total_sample_tag = sprintf('%d', params.total_samples); % Source Sample Tag

prefix = '';
if params.hyper
    prefix = 'hyper_';
    base_model_tag = [feature_tag '_' epoch_length '_' params.proc];
    model_filename = [prefix total_sample_tag '_' base_model_tag '_' params.dataset '_' params.modeltype '.mat'];
else
    base_model_tag = [feature_tag '_' epoch_length '_' params.proc];
    model_filename = [total_sample_tag '_'  base_model_tag '_' params.dataset '_' params.modeltype '.mat'];
end

% Loading the Model
fprintf('\n[INFO] Loading base model: %s\n', model_filename);
model_data = load(model_filename);

% Automatically detect the model variable (must contain "mdl" in its name)
model_vars = fieldnames(model_data);
mdl_var = model_vars(contains(model_vars, 'mdl'));

if isempty(mdl_var)
    error('[ERROR] No variable containing "mdl" found in %s', model_filename);
elseif length(mdl_var) > 1
    error('[ERROR] Multiple variables with "mdl" found in %s. Using the first one: %s', ...
        model_filename, mdl_var{1});
end

mdl_workload = model_data.(mdl_var{1});

% -------------------------------------------------------------------------
% Normalize Features (if needed) - z-score
% -------------------------------------------------------------------------
if params.only_domain_adaptation || params.do_domain_adaptation
    fprintf('[INFO] Computing normalization statistics from training data...\n');
    mu = mean(train_features);
    sigma = std(train_features);
    sigma(sigma == 0) = 1;
    
    % Source Data Normalization
    norm_val_features = (val_features - mu) ./ sigma;
    norm_test_features = (test_features - mu) ./ sigma;

    % Cross-Data Normalization
    norm_adapted_train_features = (tuning_train_features - mu) ./ sigma;
    norm_adapted_val_features = (tuning_val_features - mu) ./ sigma;
    norm_adapted_test_features = (tuning_test_features - mu) ./ sigma;
end

% -------------------------------------------------------------------------
% Domain Adaptation Only (Evaluation only)
% -------------------------------------------------------------------------
if params.only_domain_adaptation
    fprintf('[INFO] Performing Domain Adaptation (Evaluation of DA only)...\n');

    if params.hyper
        fprintf('[INFO] Evaluating Hyperparameter Tuned Model after Domain Adaptation...\n');

        hyper_model_filename = ['hyper_' total_sample_tag '_' base_model_tag '_' params.dataset '_' params.modeltype '.mat'];
        hyper_model_data = load(hyper_model_filename);
        mdl_vars = fieldnames(hyper_model_data);
        hyper_mdl_var = mdl_vars{contains(mdl_vars, 'mdl', 'IgnoreCase', true)};
        hyper_mdl_workload = hyper_model_data.(hyper_mdl_var);

        [acc2, per_class_table_cross] = eval_mdl_performance(hyper_mdl_workload, norm_adapted_val_features, tuning_val_labels, [], ...
            'DA: Hyperparameter Tuned Cross-Data Prediction', params.verbose);
        acc2 = round(acc2 * 100, 2);    % Target Accuracy (This Within Accuracy is already available in PreCalibration Results)
        export_log(params, [], acc2);
    else
        [acc2, per_class_table_cross] = eval_mdl_performance(mdl_workload, norm_adapted_val_features, tuning_val_labels, [], ...
            'DA: Cross-Data Prediction', params.verbose);
        acc2 = round(acc2 * 100, 2);    % Target Accuracy (This Within Accuracy is already available in PreCalibration Results)
        export_log(params, [], acc2);
    end

    % To ensure that all function output variables exist (avoid warning)
    acc1 = NaN;
    per_class_table_source = table();  % Or use [] if table() is undefined
end

% -------------------------------------------------------------------------
% Transfer Learning (Fine-Tuning)
% -------------------------------------------------------------------------
if params.do_transfer_learning
    fprintf('[INFO] Performing Transfer Learning...\n');

    % ---------------------- Consistency Check ----------------------------
    expected_calib = 'finetuned';
    if params.do_domain_adaptation
        expected_calib = 'finetuned_adapted';
    end
    if ~strcmp(expected_calib, params.calibration)
        warning('Calibration type in params.calibration ("%s") does not match detected configuration ("%s")', ...
            params.calibration, expected_calib);
    end

    if params.do_domain_adaptation
        fprintf('[INFO] Using Domain Adapted Cross-Data for Fine-Tuning...\n');

        if ~params.hyper
            finetune_features = norm_adapted_train_features;
            source_val_features = norm_val_features;
            source_val_tag = '[TL+DA] Fine-Tuned Domain Adapted TRAIN Dataset';
            cross_val_features = norm_adapted_val_features;
            cross_val_tag = '[TL+DA] Fine-Tuned Domain Adapted CROSS Dataset';
        else
            finetune_features = norm_adapted_train_features;
            source_val_features = norm_test_features;       % Select Test Features and Labels when using Hyperparameter Tuned
            source_val_tag = '[HYP+TL+DA] Hyperparameter Model Fine-Tuned Domain Adapted TRAIN Dataset';
            cross_val_features = norm_adapted_val_features;
            cross_val_tag = '[HYP+TL+DA] Hyperparameter Model Fine-Tuned Domain Adapted CROSS Dataset';
        end

        calib_type_tag = 'finetuned_adapted';
    else
        fprintf('[INFO] Using Not-Domain Adapted Cross-Data for Fine-Tuning...\n');

        if ~params.hyper
            finetune_features = tuning_train_features;
            source_val_features = val_features;
            source_val_tag = '[TL] Fine-Tuned TRAIN Dataset';
            cross_val_features = tuning_val_features;
            cross_val_tag = '[TL] Fine-Tuned CROSS Dataset';
        else
            finetune_features = tuning_train_features;
            source_val_features = test_features;            % Select Test Features and Labels when using Hyperparameter Tuned
            source_val_tag = '[HYP+TL] Hyperparameter Model Fine-Tuned TRAIN Dataset';
            cross_val_features = tuning_val_features;
            cross_val_tag = '[HYP+TL] Hyperparameter Model Fine-Tuned CROSS Dataset';
        end

        calib_type_tag = 'finetuned';
    end

    % Fine-tune the existing model
    new_model = fitcsvm([mdl_workload.X; finetune_features], ...
        [mdl_workload.Y; tuning_train_labels], ...
        'KernelFunction', mdl_workload.KernelParameters.Function, ...
        'BoxConstraint', mdl_workload.BoxConstraints(1));

    % Save updated model
    if params.hyper
        prefix = 'hyper_';
        base_model_tag = [feature_tag '_' epoch_length '_' params.proc];
        final_model_name = [prefix total_sample_tag '_' base_model_tag '_' params.dataset '_' calib_type_tag...
            '_'  params.modeltype '_wCross_' params.calibrationset '.mat'];
    else
        base_model_tag = [feature_tag '_' epoch_length '_' params.proc];
        final_model_name = [total_sample_tag '_'  base_model_tag '_' params.dataset '_' calib_type_tag...
            '_' params.modeltype '_wCross_' params.calibrationset '.mat'];
    end

    save(final_model_name, 'new_model');
    fprintf('[INFO] Saved new model: %s\n', final_model_name);

    if params.hyper
        source_val_labels = test_labels;
        source_val_features = test_features;
    else
        source_val_labels = val_labels;
        source_val_features = val_features;
    end

    % Evaluation

    % Evaluation of the NEW MODEL on the SOURCE DATA
    [acc1, per_class_table_source] = eval_mdl_performance(new_model, source_val_features, source_val_labels, [], source_val_tag, params.verbose);
    acc1 = round(acc1 * 100, 2);    % Source Accuracy

    % Evaluation of the NEW MODEL on the CROSS DATA
    [acc2, per_class_table_cross] = eval_mdl_performance(new_model, cross_val_features, tuning_val_labels, [], cross_val_tag, params.verbose);
    acc2 = round(acc2 * 100, 2);    % Target Accuracy

    export_log(params, acc1, acc2);

    fprintf('[INFO] Domain Adaptation / Transfer Learning Pipeline completed.\n');

end


% -------------------------------------------------------------------------
% Return values (acc1, acc2, calib_info, params)
% -------------------------------------------------------------------------

% Ensure accuracy values exist
if ~exist('acc1', 'var'), acc1 = NaN; end
if ~exist('acc2', 'var'), acc2 = NaN; end

% Collect calibration info
if params.hyper
    calib_info.samples = params.calib_samples;
    calib_info.source_total = 0.85 * params.total_samples;  % IF HYPER: Train + Validation Samples (85%)
    calib_info.ratio = round((calib_info.samples / calib_info.source_total),2) * 100;
else
    calib_info.samples = params.calib_samples;
    calib_info.source_total = 0.7 * params.total_samples;   % IF STD/NORM: Only Train Samples (70%)
    calib_info.ratio = round((calib_info.samples / calib_info.source_total),2) * 100;
end

% close all;