function [acc, acc_hyper, acc_norm, acc_hyper_norm, pct_std, pct_hyper, pct_norm, pct_hyper_norm] = train_and_eval_models(train_features, val_features, test_features, train_labels, val_labels, test_labels, opts, W_csp)

% TRAIN_AND_EVAL_MODELS - Trains and saves standard & hyperparameter tuned and normalized & hyperparametertuned SVM models.

    % Construct feature tag
    feature_str = sprintf('%d', opts.num_features); % e.g., '24', '16', etc.

    if opts.use_features && opts.use_csp
        feat_tag = [feature_str 'wCsp'];
    elseif opts.use_features
        feat_tag = feature_str;
    elseif opts.use_csp
        feat_tag = sprintf('csp_%d', opts.num_csp_filters);
    else
        error('No features specified.');
    end

    % Handle sample count prefix for saving
    if isfield(opts, 'total_samples') && opts.total_samples > 0
        sample_tag = sprintf('%d', opts.total_samples);
    else
        sample_tag = 'full';
    end

    base_filename = sprintf('%s_%s_%dsec_%s_%s', sample_tag, feat_tag, opts.epochlength, opts.proc, opts.dataset);


% -------------------------------------------------------------------------
% Standard Model
% -------------------------------------------------------------------------

    mdl = fitcsvm(train_features, train_labels, 'KernelFunction','linear');
    save([base_filename, '_model.mat'], 'mdl', 'W_csp');
    [acc, pct_std] = eval_mdl_performance(mdl, val_features, val_labels, [], 'Validation (Standard Model)', opts.verbose);
    acc = round(acc * 100, 2);
    
    % --- Hyperparameter Tuning ---
    C_vals = max(logspace(-3, 1, 5), eps);
    kernels = {'linear', 'polynomial'};
    best_C = NaN; best_kernel = ''; best_acc = -Inf;

    for k = 1:length(kernels)
        for c = 1:length(C_vals)
            mdl = fitcsvm(train_features, train_labels, ...
                'KernelFunction', kernels{k}, 'BoxConstraint', C_vals(c), 'KFold', 5);
            preds = kfoldPredict(mdl);
            accuracy = mean(preds == train_labels);
            if accuracy > best_acc
                best_C = C_vals(c);
                best_kernel = kernels{k};
                best_acc = accuracy;
            end
        end
    end

    % Retrain final hypertuned model
    mdl = fitcsvm([train_features; val_features], [train_labels; val_labels], ...
        'KernelFunction', best_kernel, 'BoxConstraint', best_C);
    save(['hyper_', base_filename, '_model.mat'], 'mdl', 'best_C', 'best_kernel', 'W_csp');
    [acc_hyper, pct_hyper] = eval_mdl_performance(mdl, test_features, test_labels, [], 'Test (Hyperparameter Tuned Standard Model)', opts.verbose);
    acc_hyper = round(acc_hyper * 100, 2);

% -------------------------------------------------------------------------
% Normalized Model
% -------------------------------------------------------------------------

    mu = mean(train_features); sigma = std(train_features); sigma(sigma==0) = 1;
    norm_train = (train_features - mu) ./ sigma;
    norm_val   = (val_features - mu) ./ sigma;
    norm_test  = (test_features - mu) ./ sigma;

    norm_mdl = fitcsvm(norm_train, train_labels, 'KernelFunction','linear');
    save([base_filename, '_norm_model.mat'], 'norm_mdl', 'W_csp');

    [acc_norm, pct_norm] = eval_mdl_performance(norm_mdl, norm_val, val_labels, [], 'Validation (Normalized Model)', opts.verbose);
    acc_norm = round(acc_norm * 100, 2);

    % --- Hyperparameter Tuning for Normalized ---
    best_C = NaN; best_kernel = ''; best_acc = -Inf;

    for k = 1:length(kernels)
        for c = 1:length(C_vals)
            mdl = fitcsvm(norm_train, train_labels, ...
                'KernelFunction', kernels{k}, 'BoxConstraint', C_vals(c), 'KFold', 5);
            preds = kfoldPredict(mdl);
            accuracy_norm = mean(preds == train_labels);
            if accuracy_norm > best_acc
                best_C = C_vals(c);
                best_kernel = kernels{k};
                best_acc = accuracy_norm;
            end
        end
    end

    % Final hypertuned normalized model
    norm_mdl = fitcsvm([norm_train; norm_val], [train_labels; val_labels], ...
        'KernelFunction', best_kernel, 'BoxConstraint', best_C);
    save(['hyper_', base_filename, '_norm_model.mat'], 'norm_mdl', 'best_C', 'best_kernel', 'W_csp');

    [acc_hyper_norm, pct_hyper_norm] = eval_mdl_performance(norm_mdl, norm_test, test_labels, [],...
        'Test (Hyperparameter Tuned Normalized Model)', opts.verbose);
    acc_hyper_norm = round(acc_hyper_norm * 100, 2);

end