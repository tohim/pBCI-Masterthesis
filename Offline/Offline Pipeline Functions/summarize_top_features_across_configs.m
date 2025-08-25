function summarize_top_features_across_configs(opts, stats_filename, top_n)

% -------------------------------------------------------------------------
% General Settings and Checks
% -------------------------------------------------------------------------
if nargin < 3, top_n = 15; end
fprintf('[INFO] Summarizing Top/Worst %d Features from: %s\n', top_n, stats_filename);

metrics = {'F_score', 'MutualInformation', 'PermutationImportance'};

if ~isfile(stats_filename)
    error('[ERROR] File does not exist: %s', stats_filename);
end

try
    [~, sheets] = xlsfinfo(stats_filename);
catch ME
    error('[ERROR] Failed to read sheet info: %s\nFile: %s', ME.message, stats_filename);
end

configs = {};
for i = 1:numel(sheets)
    s = sheets{i};
    if startsWith(s, 'F_')
        configs{end+1} = extractAfter(s, 2);
    end
end

top_summary = [];
worst_summary = [];

for i = 1:numel(configs)
    config = configs{i};
    sheet = ['F_' config];
    try
        T = readtable(stats_filename, 'Sheet', sheet);
    catch
        warning('Failed to read sheet: %s', sheet);
        continue;
    end

    combined_feats = sprintf('%dw', opts.num_features);

    % Select features                                                           
    if contains(config, combined_feats)
        feats = opts.combined_feature_names;
    elseif contains(config, 'csp')
        feats = opts.csp_feature_names;
    else
        feats = opts.handcrafted_feature_names;
    end

    % Map FeatureIDs to names
    feature_names = cell(height(T),1);
    for r = 1:height(T)
        fid = T.FeatureID(r);
        if fid > 0 && fid <= numel(feats)
            feature_names{r} = feats{fid};
        else
            feature_names{r} = 'Unknown';
        end
    end
    T.FeatureName = string(feature_names);

% -------------------------------------------------------------------------
% Normalize and Merge Feature Quality and Importance Metrics for CONFIGS
% -------------------------------------------------------------------------
    % Check and normalize each metric
    for m = 1:numel(metrics)
        metric = metrics{m};
        if ~ismember(metric, T.Properties.VariableNames)
            error('[ERROR] Metric %s not found in sheet: %s', metric, sheet);
        end
        vals = T.(metric);
        min_val = min(vals);
        max_val = max(vals);
        if max_val > min_val
            T.([metric '_norm']) = (vals - min_val) / (max_val - min_val);
        else
            warning('[WARN] Metric %s has constant values in config: %s', metric, config);
            T.([metric '_norm']) = zeros(size(vals));
        end
    end

    % Compute combined metric score
    norm_cols = strcat(metrics, '_norm');
    norm_matrix = table2array(T(:, norm_cols));
    T.MetricScore = mean(norm_matrix, 2, 'omitnan');

    % Aggregate by FeatureID
    [G, feat_ids] = findgroups(T.FeatureID);
    agg_scores = splitapply(@mean, T.MetricScore, G);
    agg_stat = splitapply(@mean, T.Cohens_d, G);

    agg_T = table(feat_ids, agg_scores, agg_stat, ...
        'VariableNames', {'FeatureID', 'MetricScore', 'Cohens_d'});
    agg_T.FeatureName = feats(feat_ids)';
    agg_T.Config = repmat({config}, height(agg_T), 1);

    sorted_top = sortrows(agg_T, 'MetricScore', 'descend');
    sorted_worst = sortrows(agg_T, 'MetricScore', 'ascend');
    top_summary = [top_summary; sorted_top(1:min(top_n, height(sorted_top)), :)];
    worst_summary = [worst_summary; sorted_worst(1:min(top_n, height(sorted_worst)), :)];
end

% Save config-wide summary
writetable(top_summary(:, {'Config','FeatureID','FeatureName','MetricScore', 'Cohens_d'}), ...
    stats_filename, 'Sheet', 'AllConfigs_TopFeatures');

writetable(worst_summary(:, {'Config','FeatureID','FeatureName','MetricScore', 'Cohens_d'}), ...
    stats_filename, 'Sheet', 'AllConfigs_WorstFeatures');


% -------------------------------------------------------------------------
% Normalize and Merge Feature Quality and Importance Metrics for DATASETS
% -------------------------------------------------------------------------
% Dataset-wide mean scores
% Collect all raw normalized rows for dataset-level summary (independent of config split)
dataset_scores = table();
for i = 1:numel(configs)
    config = configs{i};
    sheet = ['F_' config];
    try
        T = readtable(stats_filename, 'Sheet', sheet);
    catch
        warning('Failed to read sheet: %s', sheet);
        continue;
    end

    % Normalize metrics again (redundant but ensures isolation from config aggregation)
    for m = 1:numel(metrics)
        metric = metrics{m};
        if ~ismember(metric, T.Properties.VariableNames)
            warning('[WARN] Missing metric %s in %s', metric, sheet);
            T.([metric '_norm']) = zeros(height(T),1);
            continue;
        end
        vals = T.(metric);
        min_val = min(vals);
        max_val = max(vals);
        if max_val > min_val
            T.([metric '_norm']) = (vals - min_val) / (max_val - min_val);
        else
            T.([metric '_norm']) = zeros(size(vals));
        end
    end

    % Compute mean MetricScore again (separate)
    norm_cols = strcat(metrics, '_norm');
    T.MetricScore = mean(table2array(T(:, norm_cols)), 2, 'omitnan');
    dataset_scores = [dataset_scores; T];
end

% Dataset-wide mean scores
[feat_ids_ds, ~, ic_ds] = unique(dataset_scores.FeatureID);
feat_names_ds = opts.combined_feature_names(feat_ids_ds);
avg_scores_ds = accumarray(ic_ds, dataset_scores.MetricScore, [], @mean);
avg_stat_ds = accumarray(ic_ds, dataset_scores.Cohens_d, [], @mean);

total_top = sortrows(table(feat_ids_ds, feat_names_ds(:), avg_scores_ds, avg_stat_ds, ...
    'VariableNames', {'FeatureID','FeatureName','MetricScore', 'Cohens_d'}), 'MetricScore', 'descend');

total_worst = sortrows(table(feat_ids_ds, feat_names_ds(:), avg_scores_ds, avg_stat_ds, ...
    'VariableNames', {'FeatureID','FeatureName','MetricScore', 'Cohens_d'}), 'MetricScore', 'ascend');

writetable(total_top, stats_filename, 'Sheet', 'AllDatasets_TopFeatures');
writetable(total_worst, stats_filename, 'Sheet', 'AllDatasets_WorstFeatures');



% -------------------------------------------------------------------------
% Final global summary (merge AllConfigs + AllDatasets)
% -------------------------------------------------------------------------
allcfg_top = readtable(stats_filename, 'Sheet', 'AllConfigs_TopFeatures');
allds_top = readtable(stats_filename, 'Sheet', 'AllDatasets_TopFeatures');
allcfg_worst = readtable(stats_filename, 'Sheet', 'AllConfigs_WorstFeatures');
allds_worst = readtable(stats_filename, 'Sheet', 'AllDatasets_WorstFeatures');

% Top Features
[common_top_ids, idx_cfg_top, idx_ds_top] = intersect(allcfg_top.FeatureID, allds_top.FeatureID);
top_table = table;
top_table.FeatureID = common_top_ids;
top_table.FeatureName = allcfg_top.FeatureName(idx_cfg_top);
top_table.MetricScore_AllConfigs = allcfg_top.MetricScore(idx_cfg_top);
top_table.MetricScore_AllDatasets = allds_top.MetricScore(idx_ds_top);
top_table.AvgMetricScore = mean([top_table.MetricScore_AllConfigs, top_table.MetricScore_AllDatasets], 2);
top_table.Cohens_d = allcfg_top.Cohens_d(idx_cfg_top);


% Worst Features
[common_worst_ids, idx_cfg_worst, idx_ds_worst] = intersect(allcfg_worst.FeatureID, allds_worst.FeatureID);
worst_table = table;
worst_table.FeatureID = common_worst_ids;
worst_table.FeatureName = allcfg_worst.FeatureName(idx_cfg_worst);
worst_table.MetricScore_AllConfigs = allcfg_worst.MetricScore(idx_cfg_worst);
worst_table.MetricScore_AllDatasets = allds_worst.MetricScore(idx_ds_worst);
worst_table.AvgMetricScore = mean([worst_table.MetricScore_AllConfigs, worst_table.MetricScore_AllDatasets], 2);
worst_table.Cohens_d = allcfg_worst.Cohens_d(idx_cfg_worst);


% Class trend logic from class_feature_analysis 
[~, all_sheets] = xlsfinfo(stats_filename);
classwise_sheets = all_sheets(startsWith(all_sheets, 'Classwise_'));
trend_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
for i = 1:numel(classwise_sheets)
    sheet = classwise_sheets{i};
    try
        T = readtable(stats_filename, 'Sheet', sheet);
    catch
        continue;
    end
    for r = 1:height(T)
        feat = T.Feature{r};
        trend = T.ClassTrend{r};
        if ~isKey(trend_map, feat)
            trend_map(feat) = {trend};
        else
            existing = trend_map(feat);
            existing{end+1} = trend;
            trend_map(feat) = existing;
        end
    end
end

top_table.ClassTrend = arrayfun(@(i) get_consensus_trend(trend_map, top_table.FeatureName{i}), 1:height(top_table))';
worst_table.ClassTrend = arrayfun(@(i) get_consensus_trend(trend_map, worst_table.FeatureName{i}), 1:height(worst_table))';

writetable(sortrows(top_table, 'AvgMetricScore', 'descend'), stats_filename, 'Sheet', 'Total_Top_Features');
writetable(sortrows(worst_table, 'AvgMetricScore', 'ascend'), stats_filename, 'Sheet', 'Total_Worst_Features');
fprintf('[DONE] Total_Top_Features and Total_Worst_Features written.\n');

end

function trend = get_consensus_trend(trend_map, fname)
if isKey(trend_map, fname)
    t = unique(string(trend_map(fname)));
    trend = t(1);
    if numel(t) > 1, trend = "TrendUnclear"; end
else
    trend = "Unknown";
end
end
