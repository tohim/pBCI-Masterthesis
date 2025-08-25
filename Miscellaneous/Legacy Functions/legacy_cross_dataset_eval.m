function legacy_cross_dataset_eval(opts)

% CROSS_DATASET_EVAL - Pre-calibration test on a new dataset using trained models

    if opts.use_features && opts.use_csp
        feat_tag = '24wCsp';
    elseif opts.use_features
        feat_tag = '24';
    elseif opts.use_csp
        feat_tag = sprintf('csp_%d', opts.num_csp_filters);
    else
        error('No features specified.');
    end

    base_filename = sprintf('%s_%s_%s_%s', feat_tag, opts.epochlength, opts.proc, opts.dataset);
    
    % Load trained models
    model_path = [base_filename, '_model.mat'];
    hyper_model_path = ['hyper_', base_filename, '_model.mat'];

    loaded_model = load(model_path);
    model_field = fieldnames(loaded_model);
    mdl = loaded_model.(model_field{1});

    loaded_hyper = load(hyper_model_path);
    model_field_hyper = fieldnames(loaded_hyper);
    hyper_mdl = loaded_hyper.(model_field_hyper{1});

    % Load Current-Dateset Features
    [~, ~, train_features] = get_data(opts.dataset);

    % Load cross-dataset features and labels
    [cross_features, cross_labels] = get_data(opts.cross_dataset, opts);  % assumes compatible format

    % Evaluate
    eval_mdl_performance(mdl, cross_features, cross_labels, [], 'Cross Dataset (Standard Source Model)');
    eval_mdl_performance(hyper_mdl, cross_features, cross_labels, [], 'Cross Dataset (Hyper Source Model)');
    

    % Plot Feature Differences
    fprintf('\n[Plot] Feature Distribution Differences:\n');
    features_per_fig = 8;                                       % if more features then adjust amount of featperfig
    num_features = size(train_features, 2);
    num_figs = ceil(num_features / features_per_fig);

    % Replace manually as needed
    
    % Handcrafted Features
    % 24 Features 
    handcrafted_feature_names = {'Theta Power', 'Alpha Power', 'Beta Power', 'Theta Alpha Ratio', ...
        'Theta Beta Ratio', 'Alpha Beta Ratio', 'Engagement Index', 'Theta Frontal', ...
        'Theta Parietal', 'Alpha Frontal', 'Alpha Temporal', 'Alpha Parietal', ...
        'Alpha Occipital', 'Beta Frontal', 'Beta Temporal', 'Beta Parietal', ...
        'Avg Coherence', 'Theta Coherence', 'Alpha Coherence', 'Avg Mobility', ...
        'Avg Complexity', 'Avg Entropy', 'Theta Entropy', 'Alpha Entropy'};

    % 4 CSP Features
    csp_feature_names = {'CSP1_Low_Workload', 'CSP2_Low_Workload', 'CSP1_High_Workload', 'CSP2_High_Workload'};

    % Handcrafted + CSP Features
    handcrafted_csp_feature_names = [handcrafted_feature_names csp_feature_names];

    for fig_idx = 1:num_figs
        figure(fig_idx); clf;
        start_idx = (fig_idx - 1) * features_per_fig + 1;
        end_idx = min(fig_idx * features_per_fig, num_features);
        for i = start_idx:end_idx
            subplot(2, 4, i - start_idx + 1);
            histogram(train_features(:, i), 'Normalization', 'probability', 'FaceAlpha', 0.5); hold on;
            histogram(cross_features(:, i), 'Normalization', 'probability', 'FaceAlpha', 0.5);
            title(feature_names{i}); xlabel('Feature Value'); ylabel('Probability');
            legend(opts.dataset, opts.cross_dataset);
        end
        sgtitle(sprintf('Feature Distribution: %d - %d', start_idx, end_idx));
    end

end
