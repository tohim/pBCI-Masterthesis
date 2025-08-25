function run_class_feature_analysis(opts, stats_file, config_label)
%RUN_CLASS_FEATURE_ANALYSIS Performs class-wise feature comparison (Low vs High MWL)

fprintf('[INFO] Running Class-Specific Feature Analysis for: %s - %s\n', opts.dataset, config_label);

% Try loading data
try
    [~, ~, train_features, ~, ~, train_labels] = get_data(opts.dataset, opts.proc, opts);
catch ME
    fprintf('[WARNING] Could not load features for class-wise analysis: %s\n', ME.message);
    return;
end

% -------------------------------------------------------------------------
% Determine Feature Names
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

if length(feature_names) ~= n_features
    warning('[WARNING] Mismatch between feature name count (%d) and actual features (%d)', ...
        length(feature_names), n_features);
    feature_names = arrayfun(@(x) sprintf('Feature%d', x), 1:n_features, 'UniformOutput', false);
else
    fprintf('[OK] Feature count matches expected feature names.\n');
end

% -------------------------------------------------------------------------
% Class-wise Statistical Analysis
% -------------------------------------------------------------------------
cohens_d = zeros(n_features,1);
p_values = zeros(n_features,1);
selected_test = strings(n_features,1);
normality_source = strings(n_features,1);
stat_effect = strings(n_features,1);
class_trend = strings(n_features,1);

mean_low = zeros(n_features,1);
std_low = zeros(n_features,1);
mean_high = zeros(n_features,1);
std_high = zeros(n_features,1);

% Split by class
low = train_features(train_labels == 0, :);
high = train_features(train_labels == 1, :);

for i = 1:n_features
    feat_low = low(:,i);
    feat_high = high(:,i);

    % Classwise statistics
    mean_low(i) = mean(feat_low);
    std_low(i) = std(feat_low);
    mean_high(i) = mean(feat_high);
    std_high(i) = std(feat_high);

    % Determine class trend
    % ClassTrend indicates which class tends to have a higher feature value. This is not necessarily
    % an interpretation of neurophysiological relevance but a purely descriptive label to aid interpretation.
    % Of course for some features it can be, like Theta Power: often increases with cognitive load (especially frontal), 
    % same for Engagement Index (HigherInHigh)
    % Alpha Power: often decreases with attention-demanding tasks (HigherInLow)
    % Where it does not work so well: Statistical Features or CSP.
    if mean_high(i) > mean_low(i)
        class_trend(i) = "HigherInHigh";    
    elseif mean_high(i) < mean_low(i)
        class_trend(i) = "HigherInLow";
    else
        class_trend(i) = "NoDifference";
    end

    % Normality checks
    is_norm_low_ad = adtest(feat_low) == 0;
    is_norm_high_ad = adtest(feat_high) == 0;
    is_norm_low_lillie = lillietest(feat_low) == 0;
    is_norm_high_lillie = lillietest(feat_high) == 0;

    if is_norm_low_ad && is_norm_high_ad && is_norm_low_lillie && is_norm_high_lillie
        [~, p] = ttest2(feat_low, feat_high);
        selected_test(i) = "t-test";
        normality_source(i) = "BothNormal";
    elseif is_norm_low_ad && is_norm_high_ad
        [~, p] = ttest2(feat_low, feat_high);
        selected_test(i) = "t-test";
        normality_source(i) = "OnlyAD";
    else
        p = ranksum(feat_low, feat_high);
        selected_test(i) = "Mann-Whitney";
        normality_source(i) = "NotNormal";
    end
    p_values(i) = p;

    % Cohen's d
    pooled_std = sqrt((std(feat_low)^2 + std(feat_high)^2)/2);
    cohens_d(i) = (mean(feat_high) - mean(feat_low)) / (pooled_std + eps);

    % StatEffect
    if p < 0.05
        if abs(cohens_d(i)) >= 0.8
            stat_effect(i) = "sig-large";
        elseif abs(cohens_d(i)) >= 0.5
            stat_effect(i) = "sig-medium";
        elseif abs(cohens_d(i)) >= 0.2
            stat_effect(i) = "sig-small";
        else
            stat_effect(i) = "sig-tiny";
        end
    else
        stat_effect(i) = "not-sig";
    end
end

% -------------------------------------------------------------------------
% Build and Export Table
% -------------------------------------------------------------------------
DatasetCol = repmat({opts.dataset}, n_features, 1);
FeatureConfigCol = repmat({config_label}, n_features, 1);

classT = table(DatasetCol, FeatureConfigCol, feature_names', ...
               mean_low, std_low, mean_high, std_high, class_trend, ...
               p_values, cohens_d, stat_effect, selected_test, normality_source, ...
               'VariableNames', {'Dataset', 'FeatureConfig', 'Feature', ...
               'Mean_Low', 'Std_Low', 'Mean_High', 'Std_High', 'Class Trend', ...
               'pValue', 'Cohens_d', 'StatEffect', 'TestUsed', 'NormalityAgreement'});

% Sheet name per config
sheet_name = sprintf('Classwise_%s', config_label);

% Write table
writetable(classT, stats_file, 'Sheet', sheet_name, 'WriteMode', 'append');

end
