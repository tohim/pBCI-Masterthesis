function run_feature_analysis(opts, stats_file, config_label)

%RUN_FEATURE_ANALYSIS: Analyze features of a all datasets + respective feature config

fprintf('[INFO] Analyzing Features for: %s - %s...\n', opts.dataset, config_label);


% Load features and labels
try
    [~, ~, train_features, ~, ~, train_labels] = get_data(opts.dataset, opts.proc, opts);
catch ME
    fprintf('[WARNING] Could not load features: %s\n', ME.message);
    return;
end


% -------------------------------------------------------------------------
% Feature Name Mapping
% -------------------------------------------------------------------------
if opts.use_features && opts.use_csp
    feature_names = opts.combined_feature_names;
elseif opts.use_features
    feature_names = opts.handcrafted_feature_names;
elseif opts.use_csp
    feature_names = opts.csp_feature_names;
else
    error('Unknown feature configuration (neither handcrafted nor CSP)');
end

[~, n_features] = size(train_features);
feature_ids = (1:n_features)';

if length(feature_names) ~= n_features
    warning('[WARNING] Mismatch between feature name count (%d) and actual features (%d)', ...
        length(feature_names), n_features);
    feature_names = arrayfun(@(x) sprintf('Feature%d', x), 1:n_features, 'UniformOutput', false);
else
    fprintf('[OK] Feature count matches expected feature names.\n');
end

% -------------------------------------------------------------------------
% Parameters/ Stats and Initializations
% -------------------------------------------------------------------------
p_vals = NaN(n_features, 1);
cohens_d = NaN(n_features, 1);
selected_test = strings(n_features, 1);
normality_agreement = strings(n_features, 1);

% Basic Statistics
means = mean(train_features);
stds = std(train_features);

% -------------------------------------------------------------------------
% Per-Feature Analysis
% -------------------------------------------------------------------------
for k = 1:n_features
    feat = train_features(:, k);
    group1 = feat(train_labels == 0);
    group2 = feat(train_labels == 1);

    % Normality checks
    % Anderson-Darling (more sensitive in tails of distribution)
    is_norm1_ad = adtest(group1) == 0;
    is_norm2_ad = adtest(group2) == 0;

    % Kolmogorov Smirnov-Lielliefors
    is_norm1_lillie = lillietest(group1) == 0;
    is_norm2_lillie = lillietest(group2) == 0;

    % Total Features Anylsis: How well does a feature behave in general in
    % relation to workload / how useful is it for classification.

    % p-vals: Measure statistical significance of difference between class
    % 0 and 1 for a feature = "Which feature seprates classes well statistically
    % p < 0.05 => strong evidence that the feature values contribute substantially to assess Mental Workload
    % Can explain which features are truly discriminative STANDALONE predictors
    % Effect Size gives a perspective on how strong they contribute 

    % Features can be ranked non-significantly, but still contribute value
    % information for the classification based on interaction effects with
    % other features or due to non-linearity components.

    % Choose test based on agreement: 
    if is_norm1_ad && is_norm2_ad && is_norm1_lillie && is_norm2_lillie
        [~, p_vals(k)] = ttest2(group1, group2);
        selected_test(k) = "t-test";
        normality_agreement(k) = "BothNormal";
    elseif is_norm1_ad && is_norm2_ad
        [~, p_vals(k)] = ttest2(group1, group2);
        selected_test(k) = "t-test";
        normality_agreement(k) = "OnlyADNormal";
        warning('[%s] Feature %s: Only AD test passed, Lilliefors failed. Still using t-test.', ...
                opts.dataset, feature_names{k});
    else
        p_vals(k) = ranksum(group1, group2);
        selected_test(k) = "Mann-Whitney";
        normality_agreement(k) = "NotNormal";
    end

% -------------------------------------------------------------------------
% Cohen's d
% -------------------------------------------------------------------------
% Adds practical discriminative power. Quantifies how far apart 2
% distributions are (in Std). Adds practical importance measure to
% significant p-values. But does not account for feature distribution or
% classification utility.
% 0.2 - 0.3 -> Small Effect (Weak Predictor)
% 0.5       -> Medium Effect (Moderate Predictor)
% 0.8+      -> Large Effect (Strong Predictor)

    pooled_std = sqrt(((numel(group1)-1)*var(group1) + (numel(group2)-1)*var(group2)) / ...
                      (numel(group1) + numel(group2) - 2));
    cohens_d(k) = abs(mean(group1) - mean(group2)) / (pooled_std + eps);

end

% Note: T-Test/ Mann-Whitney-U Test together with Cohen's d can be used as
% early warning indicators. They help:
% - detect completely uninformative features (bad across all metrics)
% - report statistical justifications in thesis/ add scientific credibility 
% - spot potential false positives or false negatives in model-based importance
% Features that perform poorly on t-test & cohen's d but high on MI or
% PermImp may be synergistic or non-linear features, so still important.

% Combined Statistical Significance + Effect Size Category
StatEffect = strings(n_features, 1);

for i = 1:n_features
    if p_vals(i) >= 0.05
        StatEffect(i) = "not-sig";  % not significant
    elseif cohens_d(i) >= 0.8
        StatEffect(i) = "sig-large";
    elseif cohens_d(i) >= 0.5
        StatEffect(i) = "sig-medium";
    elseif cohens_d(i) >= 0.2
        StatEffect(i) = "sig-small";
    else
        StatEffect(i) = "sig-tiny";
    end
end


% -------------------------------------------------------------------------
% ANOVA F-score
% -------------------------------------------------------------------------
% Measures the between-class vs. within-class variance 
% Works well with numerical and normally distributed features (not
% necessarily given here). Strong for measuring discriminability.
f_scores = compute_anova_f(train_features, train_labels);

% -------------------------------------------------------------------------
% Mutual Information (MI)
% -------------------------------------------------------------------------
% Measures amount of shared information between a feature and the class labels.
% => Measures the direct feature <-> label correlation (model independent).
% But doesnt capture feature interactions. Measures how much knowing a
% feature reduces uncertainty about the class. Non-linear & Non-parametric.
mi_scores = compute_mutual_info(train_features, train_labels);

% -------------------------------------------------------------------------
% Permutation Importance (default: accuracy drop from SVM)
% -------------------------------------------------------------------------
% Assesses the impact of each feature on the model's performance.
% Measures how much the accuracy decreases when u randomly shuffle a feature's values.
% Concept of Permutation Importance per se is Model-agnostic, but when applied it
% is model-dependent - cannot be computed from only feature-label correlation (unlikey Mutual Information).
mdl = fitcsvm(train_features, train_labels, 'KernelFunction', 'linear');    % (Re)training a new model needed to check pure feature importance
perm_importance = compute_permutation_importance(train_features, train_labels, mdl);


% -------------------------------------------------------------------------
% Build Table
% -------------------------------------------------------------------------
% Context Columns
DatasetCol = repmat({opts.dataset}, n_features, 1);
ConfigCol = repmat({config_label}, n_features, 1);
FeatureNameCol = feature_names(:);  % Ensure column vector

% Build table
T = table(DatasetCol, ConfigCol, feature_ids, FeatureNameCol, means', stds', ...
    selected_test, normality_agreement, p_vals, cohens_d, StatEffect, ...
    f_scores', mi_scores', perm_importance', ...
    'VariableNames', {'Dataset', 'FeatureConfig', 'FeatureID', ...
    'FeatureName', 'Mean', 'Std', 'StatTest', 'NormalityAgreement', ...
    'P_value', 'Cohens_d', 'StatEffect', 'F_score', ...
    'MutualInformation', 'PermutationImportance'});

T = sortrows(T, 'F_score', 'descend');

% Determine sheet name (by config only)
% Merge by Configuration (One Sheet per Configuration) => all datasets in 1 sheet
% = Comparing cross-dataset generalizability of feature importance ("Are certain features consistently strong across datasets?")
% Merge by Configuration (F_24, F_Csp, F_24wCsp): Better matches my real goal of selecting features that are:
% - consistently useful across datasets
% - more likely to generalize
% - suitable for cross-data workload classification
sheet_name = sprintf('F_%s', config_label);

% Append to config-based sheet (header will only appear once if you run clean each time)
writetable(T, stats_file, 'Sheet', sheet_name, 'WriteMode', 'append');

fprintf('[SAVED] → %s (sheet: %s)\n', stats_file, sheet_name);

end



% -------------------------------------------------------------------------
% HELPER FUNCTIONS
% -------------------------------------------------------------------------

function mi_scores = compute_mutual_info(X, y)
    n_features = size(X, 2);
    mi_scores = zeros(1, n_features);
    for k = 1:n_features
        xi = X(:, k);
        mi_scores(k) = mutualinfo_discrete(xi, y);
    end
end

function mi = mutualinfo_discrete(x, y)
    % Use fixed bin count to ensure stability
    nbins = 10;
    x_disc = discretize(x, linspace(min(x), max(x), nbins + 1));
    
    % Ensure y is binary
    y_disc = y;

    joint = accumarray([x_disc, y_disc + 1], 1); % +1 for MATLAB indexing
    joint_prob = joint / sum(joint(:));
    px = sum(joint_prob, 2);
    py = sum(joint_prob, 1);

    mi = 0;
    for a = 1:size(joint_prob, 1)
        for b = 1:size(joint_prob, 2)
            if joint_prob(a, b) > 0
                mi = mi + joint_prob(a, b) * log2(joint_prob(a, b) / (px(a) * py(b)));
            end
        end
    end
end


function perm_importance = compute_permutation_importance(X, y, model, n_repeats)
    if nargin < 4
        n_repeats = 10;
    end

    n_features = size(X, 2);
    perm_importance = zeros(1, n_features);
    y_pred = predict(model, X);
    acc_baseline = mean(y_pred == y);

    for f = 1:n_features
        acc_drops = zeros(1, n_repeats);
        for r = 1:n_repeats
            X_perm = X;
            X_perm(:, f) = X(randperm(size(X, 1)), f);
            y_pred_perm = predict(model, X_perm);
            acc_perm = mean(y_pred_perm == y);
            acc_drops(r) = acc_baseline - acc_perm;
        end
        perm_importance(f) = mean(acc_drops);
    end
end



% function mi_scores = compute_mutual_info(X, y)
%     n_features = size(X, 2);
%     mi_scores = zeros(1, n_features);
%     for k = 1:n_features
%         xi = X(:, k);
%         mi_scores(k) = mutualinfo_discrete(xi, y);
%     end
% end
% 
% function mi = mutualinfo_discrete(x, y)
%     nbins = round(sqrt(length(x)));
%     x_disc = discretize(x, nbins);
%     y_disc = discretize(y, 2);  % Binary labels
% 
%     joint = accumarray([x_disc, y_disc], 1);
%     joint_prob = joint / sum(joint(:));
%     px = sum(joint_prob, 2);
%     py = sum(joint_prob, 1);
% 
%     mi = 0;
%     for a = 1:size(joint_prob, 1)
%         for b = 1:size(joint_prob, 2)
%             if joint_prob(a, b) > 0
%                 mi = mi + joint_prob(a, b) * log2(joint_prob(a, b) / (px(a) * py(b)));
%             end
%         end
%     end
% end
% 
% function perm_importance = compute_permutation_importance(X, y, model)
%     n_features = size(X, 2);
%     perm_importance = zeros(1, n_features);
% 
%     y_pred = predict(model, X);
%     acc_baseline = mean(y_pred == y);
% 
%     for f = 1:n_features
%         X_perm = X;
%         X_perm(:, f) = X(randperm(size(X, 1)), f);
%         y_pred_perm = predict(model, X_perm);
%         acc_perm = mean(y_pred_perm == y);
%         perm_importance(f) = acc_baseline - acc_perm;
%     end
% end
