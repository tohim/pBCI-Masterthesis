% VERSION SPECIFIC FINAL SUMMARY AGGREGATOR ACROSS ALL FOLDERS

% -------------------------------------------------------------------------
% SETTINGS
% -------------------------------------------------------------------------
base_pipeline_path     = fullfile('E:', 'SchuleJobAusbildung', 'HTW', 'MasterThesis', ...
    'Code', 'Matlab', 'Data', 'AutoPipeline', 'v1');                                            % !! CHANGE VERSION !!                                                
base_calibration_path  = fullfile('E:', 'SchuleJobAusbildung', 'HTW', 'MasterThesis', ...
    'Code', 'Matlab', 'Data', 'AutoCalibration', 'v1');                                         % !! CHANGE VERSION !!      

% Get all calibration folders
calib_folders = dir(fullfile(base_calibration_path, '*samples_*Pct_Calib'));
calib_folders = calib_folders([calib_folders.isdir]);

% Init result containers
AllWithin = table();
AllCross = table();
AllDeltaSummary = table();
AllTWithin = table();

% -------------------------------------------------------------------------
% LOOP OVER ALL CALIBRATION FOLDERS
% -------------------------------------------------------------------------
for i = 1:length(calib_folders)
    folder_name = calib_folders(i).name;
    fprintf('\n[INFO] Processing folder: %s\n', folder_name);

    tokens = regexp(folder_name, '(\d+)samples_(\d+)Pct_Calib', 'tokens');
    if isempty(tokens), continue; end

    % Calibration sample mapping + percentage
    calib_map = containers.Map;
    calib_map('1000_10') = [70, 10];    calib_map('1000_15') = [106, 15];
    calib_map('1000_20') = [140, 20];   calib_map('1000_25') = [176, 25];
    calib_map('1000_30') = [210, 30];   %calib_map('2000_10') = [140, 10];
%     calib_map('2000_15') = [210, 15];   calib_map('2000_20') = [280, 20];
%     calib_map('2000_25') = [350, 25];   calib_map('2000_30') = [420, 30];
%     calib_map('3000_10') = [210, 10];   calib_map('4000_10') = [280, 10];

    sample_count = str2double(tokens{1}{1});
    pct_val      = str2double(tokens{1}{2});

    if sample_count < 1000
        fprintf('\n[INFO] Skipping test-sample folder: %s\n', folder_name);
        continue;
    end

    lookup_key = sprintf('%d_%d', sample_count, pct_val);
    if ~isKey(calib_map, lookup_key)
        warning('[!] Missing calib sample mapping for: %s\n', lookup_key);
        continue;
    end
    calib_info  = calib_map(lookup_key); % [count, percent]
    calib_count = calib_info(1);
    calib_pct   = calib_info(2);

    pipeline_result_file = fullfile(base_pipeline_path, folder_name, ...
        sprintf('PreCalib_%d_Samples_Within_Cross_Results.xlsx', sample_count));
    stats_file = fullfile(base_calibration_path, folder_name, ...
        sprintf('AfterCalib_%dTotal_%dCalib_Stats.xlsx', sample_count, calib_count));
    pre_file = fullfile(base_calibration_path, folder_name, 'PreCalib_Cross_Results.mat');
    post_file = fullfile(base_calibration_path, folder_name, 'PostCalib_Cross_Results.mat');

    try
        T_total = readtable(pipeline_result_file, 'Sheet', 'Total');
        T_within = T_total(strcmp(T_total.TARGET, 'Within'), :);
        
        % Get Source Model Name
        tokens = cellfun(@(x) split(string(x), ' '), T_within.SOURCE, 'UniformOutput', false);
        T_within.Model = strings(height(T_within), 1); % Preallocate

        for k = 1:height(T_within)
            toks = tokens{k};

            if numel(toks) >= 2 && toks(end) == "NORM" && toks(end-1) == "HYPER"
                T_within.Model(k) = "HYPER NORM";
            else
                T_within.Model(k) = toks(end);  % take last token
            end
        end
        
        % Get Source Config Name
        T_within.Config = repmat("Unknown", height(T_within), 1);                           % !! CHANGE CONFIG NAMES IF DIFFERENT VERSION !!  
        for k = 1:height(T_within)
            src = string(T_within.SOURCE(k));
            if contains(src, "25wCsp")
                T_within.Config(k) = "25wCsp";
            elseif contains(src, "csp")
                T_within.Config(k) = "Csp";
            elseif contains(src, "25")
                T_within.Config(k) = "25";
            end
        end

        T_within.Dataset = extractBefore(T_within.SOURCE, ' ');
        T_within.TotalSamples = repmat(sample_count, height(T_within), 1);
        T_within.CalibSamples = repmat(calib_count, height(T_within), 1);
        AllTWithin = [AllTWithin; T_within];

        % Load Pre/Post
        T_pre = load(pre_file);   
        T_pre = T_pre.T_pre_cross_accuracies;
        T_post = load(post_file); 
        T_post = T_post.T_post_cross_accuracies;

        % Add required fields for proper tracking
        T_pre.CalibrationType = repmat("None", height(T_pre), 1);
        T_pre.Stage = repmat("Pre", height(T_pre), 1);
        T_post.Stage = repmat("Post", height(T_post), 1);

        Cross = [T_pre; T_post];
        Cross.TotalSamples = repmat(sample_count, height(Cross), 1);
        Cross.CalibSamples = repmat(calib_count, height(Cross), 1);
        Cross.CalibPercent = repmat(calib_pct, height(Cross), 1);
        Cross.Dataset = extractBefore(Cross.SOURCE, ' ');

        % Correct Model names for Cross
        tokens_cross = cellfun(@(x) split(string(x), ' '), Cross.SOURCE, 'UniformOutput', false);
        Cross.Model = strings(height(Cross), 1); % Preallocate

        for k = 1:height(Cross)
            toks = tokens_cross{k};
            if numel(toks) >= 2 && toks(end) == "NORM" && toks(end-1) == "HYPER"
                Cross.Model(k) = "HYPER NORM";
            else
                Cross.Model(k) = toks(end);
            end
        end

        AllCross = [AllCross; Cross];

        % Optional: Delta Summary (but without fake Dataset)
        [~, sheets] = xlsfinfo(stats_file);
        if ismember('Delta_Summary', sheets)
            T_delta_summary = readtable(stats_file, 'Sheet', 'Delta_Summary');
            if ~isempty(T_delta_summary)
                T_delta_summary.TotalSamples = repmat(sample_count, height(T_delta_summary), 1);
                T_delta_summary.CalibSamples = repmat(calib_count, height(T_delta_summary), 1);
                T_delta_summary.CalibPercent = repmat(calib_pct, height(T_delta_summary), 1);
                AllDeltaSummary = [AllDeltaSummary; T_delta_summary];
            end
        end

        % PER-FOLDER WITHIN AGGREGATION
        W_config = varfun(@mean, T_within, 'InputVariables', 'ACCURACY', ...
            'GroupingVariables', {'Dataset','Config','TotalSamples','CalibSamples'});
        W_config.Type = repmat("By_Config", height(W_config), 1);

        W_model = varfun(@mean, T_within, 'InputVariables', 'ACCURACY', ...
            'GroupingVariables', {'Dataset','Model','TotalSamples','CalibSamples'});
        W_model.Type = repmat("By_Model", height(W_model), 1);
        W_model.Config = W_model.Model; W_model = removevars(W_model, 'Model');

        W_combined = [W_config(:, {'Dataset','Config','Type','mean_ACCURACY','TotalSamples','CalibSamples'}); ...
                      W_model(:, {'Dataset','Config','Type','mean_ACCURACY','TotalSamples','CalibSamples'})];
        W_combined.Properties.VariableNames{'mean_ACCURACY'} = 'Mean_Within';
        AllWithin = [AllWithin; W_combined];

    catch ME
        warning('Skipping %s due to missing files: %s\n', folder_name, ME.message);
        continue;
    end
end

% -------------------------------------------------------------------------
% TRUE OVERALL WITHIN MEANS
% -------------------------------------------------------------------------
W_all = varfun(@mean, AllTWithin, 'InputVariables', 'ACCURACY', ...
    'GroupingVariables', {'Dataset','TotalSamples'});
W_all.Type = repmat("All", height(W_all), 1);
W_all.Config = repmat("All", height(W_all), 1);
W_all.CalibSamples = repmat(0, height(W_all), 1);
W_all.Properties.VariableNames{'mean_ACCURACY'} = 'Mean_Within';
AllWithin = [W_all(:, {'Dataset','Config','Type','Mean_Within','TotalSamples','CalibSamples'}); AllWithin];
summary_within = W_all(:, {'Dataset','Mean_Within','TotalSamples'});


% -------------------------------------------------------------------------
% SUMMARY: MEAN WITHIN ACCURACIES - PER CONFIGURATION
% -------------------------------------------------------------------------
summary_within_configs = varfun(@mean, AllTWithin, ...
    'InputVariables', 'ACCURACY', ...
    'GroupingVariables', {'Config','TotalSamples'});

summary_within_configs.Properties.VariableNames{'mean_ACCURACY'} = 'Mean_Within_Config';

% Optional: Sort it cleanly
summary_within_configs = sortrows(summary_within_configs, {'Config','TotalSamples'});




% -------------------------------------------------------------------------
% SUMMARY: MEANS WITHIN ACCURACIES - PER MODELTYPE
% -------------------------------------------------------------------------
summary_within_modeltypes = groupsummary(AllTWithin, {'Model','TotalSamples'}, {'mean','numel'}, 'ACCURACY');

% Sort nicely
summary_within_modeltypes = sortrows(summary_within_modeltypes, {'TotalSamples','Model'});


% === NEW: Summary of Model Types (ONLY 25wCsp) ===
Only25wCsp_Within = AllTWithin(AllTWithin.Config == "25wCsp", :);
summary_modeltypes_25wCsp = groupsummary(Only25wCsp_Within, ...
    {'Model','TotalSamples'}, {'mean','numel'}, 'ACCURACY');

summary_modeltypes_25wCsp.Properties.VariableNames{'mean_ACCURACY'} = 'Mean_Accuracy';
summary_modeltypes_25wCsp.Properties.VariableNames{'GroupCount'} = 'N';
summary_modeltypes_25wCsp = sortrows(summary_modeltypes_25wCsp, {'TotalSamples','Mean_Accuracy'}, 'descend');


% === Mean and STD accuracy per model type for 25wCsp only ===
modeltype_summary_25wCsp = groupsummary( ...
    AllTWithin(strcmp(AllTWithin.Config, "25wCsp"), :), ...
    "Model", {"mean", "std"}, "ACCURACY");

% Rename columns for clarity
modeltype_summary_25wCsp.Properties.VariableNames{'mean_ACCURACY'} = 'Mean_Accuracy';
modeltype_summary_25wCsp.Properties.VariableNames{'std_ACCURACY'} = 'Std_Accuracy';

% Optional: sort descending
modeltype_summary_25wCsp = sortrows(modeltype_summary_25wCsp, 'Mean_Accuracy', 'descend');



% -------------------------------------------------------------------------
% CROSS SUMMARIES (Pre and Post separated)
% -------------------------------------------------------------------------
summary_cross = varfun(@mean, AllCross, ...
    'InputVariables', 'ACCURACY', ...
    'GroupingVariables', {'Dataset','CalibrationType','Stage','TotalSamples','CalibSamples','CalibPercent'});
summary_cross.Properties.VariableNames{'mean_ACCURACY'} = 'Avg_Accuracy';

% Use only 25wCsp, post-stage
cross_25wCsp = AllCross(strcmp(AllCross.Config,'25wCsp') & ...
                        strcmp(AllCross.Stage,'Post') & ...
                        ismember(AllCross.CalibrationType,{'adapted','finetuned','finetuned_adapted'}), :);

% Mean ACCURACY per Dataset × CalibPercent × CalibrationType
G = groupsummary(cross_25wCsp, {'Dataset','CalibPercent','CalibrationType'}, 'mean','ACCURACY');
% -> G.mean_ACCURACY is the per-dataset mean for that (CalibPercent, CalibrationType)

% Aggregate over datasets: mean and std of those per-dataset means
S = groupsummary(G, {'CalibPercent','CalibrationType'}, {'mean','std'}, 'mean_ACCURACY');
S.Properties.VariableNames{'mean_mean_ACCURACY'} = 'Delta_Mean';
S.Properties.VariableNames{'std_mean_ACCURACY'}  = 'Delta_Std';

% Pivot into wide format with Mean and Std side-by-side for each type
calib_types = {'adapted','finetuned','finetuned_adapted'};
Summary_25wCsp = table();
Summary_25wCsp.CalibPercent = unique(S.CalibPercent);

for i = 1:numel(calib_types)
    type = calib_types{i};
    rows  = strcmp(S.CalibrationType, type);
    Ttype = S(rows, {'CalibPercent','Delta_Mean','Delta_Std'});

    [~, idx] = ismember(Summary_25wCsp.CalibPercent, Ttype.CalibPercent);
    m = nan(height(Summary_25wCsp),1);
    s = nan(height(Summary_25wCsp),1);
    ok = idx > 0;
    m(ok) = Ttype.Delta_Mean(idx(ok));
    s(ok) = Ttype.Delta_Std(idx(ok));

    Summary_25wCsp.(sprintf('%s_Mean', type)) = m;
    Summary_25wCsp.(sprintf('%s_Std',  type)) = s;
end


% -------------------------------------------------------------------------
% SUMMARY CROSS: by Configs + CalibPercent
% -------------------------------------------------------------------------

summary_cross_configs = groupsummary(AllCross, {'Config','CalibPercent','TotalSamples'}, {'mean','numel'}, 'ACCURACY');

if ismember('nummissing_ACCURACY', summary_cross_configs.Properties.VariableNames)
    summary_cross_configs = removevars(summary_cross_configs, 'nummissing_ACCURACY');
end

% Clean column names
summary_cross_configs.Properties.VariableNames{'Config'} = 'Config';
summary_cross_configs.Properties.VariableNames{'CalibPercent'} = 'CalibPercent';
summary_cross_configs.Properties.VariableNames{'TotalSamples'} = 'TotalSamples';
summary_cross_configs.Properties.VariableNames{'mean_ACCURACY'} = 'Mean_Cross';
summary_cross_configs.Properties.VariableNames{'GroupCount'} = 'Group_Count';

% Unique CalibPercents
calib_percents = unique(summary_cross_configs.CalibPercent);

Top_cross_configs = table(); % Initialize empty

for i = 1:length(calib_percents)
    this_pct = calib_percents(i);
    subset = summary_cross_configs(summary_cross_configs.CalibPercent == this_pct, :);
    subset = sortrows(subset, 'Mean_Cross', 'descend');
    Top_cross_configs = [Top_cross_configs; subset(1,:)];
end

summary_cross_configs = Top_cross_configs;
summary_cross_configs = sortrows(summary_cross_configs, {'CalibPercent','Mean_Cross'}, {'ascend','descend'});




% -------------------------------------------------------------------------
% SUMMARY CROSS: by Modeltypes + CalibPercent
% -------------------------------------------------------------------------

summary_cross_modeltypes = groupsummary(AllCross, {'Model','CalibPercent','TotalSamples'}, {'mean','numel'}, 'ACCURACY');

if ismember('nummissing_ACCURACY', summary_cross_modeltypes.Properties.VariableNames)
    summary_cross_modeltypes = removevars(summary_cross_modeltypes, 'nummissing_ACCURACY');
end

% Clean column names
summary_cross_modeltypes.Properties.VariableNames{'Model'} = 'Model';
summary_cross_modeltypes.Properties.VariableNames{'CalibPercent'} = 'CalibPercent';
summary_cross_modeltypes.Properties.VariableNames{'TotalSamples'} = 'TotalSamples';
summary_cross_modeltypes.Properties.VariableNames{'mean_ACCURACY'} = 'Mean_Cross';
summary_cross_modeltypes.Properties.VariableNames{'GroupCount'} = 'Group_Count';

calib_percents = unique(summary_cross_modeltypes.CalibPercent);

Top2_cross_models = table(); % Initialize

for i = 1:length(calib_percents)
    this_pct = calib_percents(i);
    subset = summary_cross_modeltypes(summary_cross_modeltypes.CalibPercent == this_pct, :);
    subset = sortrows(subset, 'Mean_Cross', 'descend');
    Top2_cross_models = [Top2_cross_models; subset(1:min(2,height(subset)), :)];
end

summary_cross_modeltypes = Top2_cross_models;
summary_cross_modeltypes = sortrows(summary_cross_modeltypes, {'CalibPercent','Mean_Cross'}, {'ascend','descend'});

% -------------------------------------------------------------------------
% SUMMARY OF DELTA IMPROVEMENT ACROSS CALIBRATION TYPES
% -------------------------------------------------------------------------
summary_means = groupsummary(AllDeltaSummary, {'CalibPercent','CalibrationType'}, 'mean', 'Mean');
summary_stds  = groupsummary(AllDeltaSummary, {'CalibPercent','CalibrationType'}, 'std', 'Mean');
summary_means.Properties.VariableNames{'mean_Mean'} = 'Mean_Delta';
summary_stds.Properties.VariableNames{'std_Mean'}   = 'Std_Delta';

calib_types = unique(summary_means.CalibrationType);
Summary_Delta_All = table();
Summary_Delta_All.CalibPercent = unique(summary_means.CalibPercent);

for i = 1:length(calib_types)
    calib = calib_types{i};
    temp_mean = summary_means(strcmp(summary_means.CalibrationType, calib), :);
    [~, idx_mean] = ismember(Summary_Delta_All.CalibPercent, temp_mean.CalibPercent);
    mean_values = nan(height(Summary_Delta_All), 1);
    valid_idx_mean = idx_mean > 0;
    mean_values(valid_idx_mean) = temp_mean.Mean_Delta(idx_mean(valid_idx_mean));
    Summary_Delta_All.(sprintf('%s_Mean', calib)) = mean_values;

    temp_std = summary_stds(strcmp(summary_stds.CalibrationType, calib), :);
    [~, idx_std] = ismember(Summary_Delta_All.CalibPercent, temp_std.CalibPercent);
    std_values = nan(height(Summary_Delta_All), 1);
    valid_idx_std = idx_std > 0;
    std_values(valid_idx_std) = temp_std.Std_Delta(idx_std(valid_idx_std));
    Summary_Delta_All.(sprintf('%s_Std', calib)) = std_values;
end

summary_delta_overall = groupsummary(AllDeltaSummary, 'CalibrationType', {'mean','median','std'}, 'Mean');
G_overall = findgroups(AllDeltaSummary.CalibrationType);
N_overall = splitapply(@numel, AllDeltaSummary.Mean, G_overall);
summary_delta_overall.N = N_overall;
summary_delta_overall.Properties.VariableNames{'mean_Mean'}   = 'Mean_Delta';
summary_delta_overall.Properties.VariableNames{'median_Mean'} = 'Median_Delta';
summary_delta_overall.Properties.VariableNames{'std_Mean'}    = 'Std_Delta';
summary_delta_overall = sortrows(summary_delta_overall, 'CalibrationType');


% -------------------------------------------------------------------------
% SUMMARY: DELTA MEAN & STD PER DATASET × CALIBPERCENT × CALIBRATIONTYPE
% -------------------------------------------------------------------------
% Use only POST-calibration accuracies to compute delta
T_post = AllCross(AllCross.Stage == "Post", :);

% Only valid calib types
valid_rows = ismember(T_post.CalibrationType, ...
    {'adapted', 'finetuned', 'finetuned_adapted'});
T = T_post(valid_rows, :);

% Compute mean and std for each Dataset × CalibPercent × CalibrationType
mean_table = groupsummary(T, {'Dataset','CalibPercent','CalibrationType'}, 'mean', 'ACCURACY');
std_table  = groupsummary(T, {'Dataset','CalibPercent','CalibrationType'}, 'std',  'ACCURACY');

% Rename for clarity
mean_table.Properties.VariableNames{'mean_ACCURACY'} = 'Delta_Mean';
std_table.Properties.VariableNames{'std_ACCURACY'}   = 'Delta_Std';

% Initialize wide summary table
datasets = unique(mean_table.Dataset);
calib_percents = unique(mean_table.CalibPercent);
calib_types = unique(mean_table.CalibrationType);

[DD, PP] = ndgrid(datasets, calib_percents);
Summary_Delta_ByDatasetPct = table();
Summary_Delta_ByDatasetPct.Dataset = DD(:);
Summary_Delta_ByDatasetPct.CalibPercent = PP(:);

% Fill columns per calibration type
for i = 1:length(calib_types)
    calib = calib_types{i};

    % Mean
    temp_mean = mean_table(strcmp(mean_table.CalibrationType, calib), :);
    key = strcat(temp_mean.Dataset, "_", string(temp_mean.CalibPercent));
    current_keys = strcat(Summary_Delta_ByDatasetPct.Dataset, "_", string(Summary_Delta_ByDatasetPct.CalibPercent));
    [~, idx] = ismember(current_keys, key);
    col = nan(height(Summary_Delta_ByDatasetPct), 1);
    valid = idx > 0;
    col(valid) = temp_mean.Delta_Mean(idx(valid));
    Summary_Delta_ByDatasetPct.(sprintf('%s_Mean', calib)) = col;

    % Std
    temp_std = std_table(strcmp(std_table.CalibrationType, calib), :);
    key = strcat(temp_std.Dataset, "_", string(temp_std.CalibPercent));
    [~, idx] = ismember(current_keys, key);
    col = nan(height(Summary_Delta_ByDatasetPct), 1);
    valid = idx > 0;
    col(valid) = temp_std.Delta_Std(idx(valid));
    Summary_Delta_ByDatasetPct.(sprintf('%s_Std', calib)) = col;
end

Summary_Delta_ByDatasetPct = sortrows(Summary_Delta_ByDatasetPct, {'Dataset','CalibPercent'});


% -------------------------------------------------------------------------
% Condensed: BEST Calibration Type per Dataset × CalibPercent
% -------------------------------------------------------------------------
BestCalib_ByDatasetPct = table();
BestCalib_ByDatasetPct.Dataset = Summary_Delta_ByDatasetPct.Dataset;
BestCalib_ByDatasetPct.CalibPercent = Summary_Delta_ByDatasetPct.CalibPercent;
BestCalib_ByDatasetPct.BestType = strings(height(Summary_Delta_ByDatasetPct),1);
BestCalib_ByDatasetPct.BestMean = zeros(height(Summary_Delta_ByDatasetPct),1);
BestCalib_ByDatasetPct.BestStd = zeros(height(Summary_Delta_ByDatasetPct),1);

% Calibration types to check
calib_types = {'adapted', 'finetuned', 'finetuned_adapted'};

% For each row, determine best calibration type by highest Mean
for i = 1:height(Summary_Delta_ByDatasetPct)
    means = zeros(1, length(calib_types));
    stds  = zeros(1, length(calib_types));
    for j = 1:length(calib_types)
        type = calib_types{j};
        mean_col = sprintf('%s_Mean', type);
        std_col  = sprintf('%s_Std', type);
        means(j) = Summary_Delta_ByDatasetPct{i, mean_col};
        stds(j)  = Summary_Delta_ByDatasetPct{i, std_col};
    end

    [best_val, best_idx] = max(means);
    BestCalib_ByDatasetPct.BestType(i) = calib_types{best_idx};
    BestCalib_ByDatasetPct.BestMean(i) = best_val;
    BestCalib_ByDatasetPct.BestStd(i)  = stds(best_idx);
end

BestCalib_ByDatasetPct = sortrows(BestCalib_ByDatasetPct, {'Dataset','CalibPercent'});


% -------------------------------------------------------------------------
% Condensed: Best Calibration Type per CONFIG × CalibPercent
% -------------------------------------------------------------------------

% Use only Post-calibration rows for this summary
PostCross = AllCross(strcmp(AllCross.Stage, "Post"), :);

% Group and compute mean/std accuracy per Config × CalibPercent × CalibType
mean_table = groupsummary(PostCross, {'Config','CalibPercent','CalibrationType'}, 'mean', 'ACCURACY');
std_table  = groupsummary(PostCross, {'Config','CalibPercent','CalibrationType'}, 'std',  'ACCURACY');

% Rename columns for clarity
mean_table.Properties.VariableNames{'mean_ACCURACY'} = 'Mean_Accuracy';
std_table.Properties.VariableNames{'std_ACCURACY'}   = 'Std_Accuracy';

% Join mean and std into one table
stats_table = outerjoin(mean_table, std_table, ...
    'Keys', {'Config','CalibPercent','CalibrationType'}, 'MergeKeys', true);

% Group by Config and CalibPercent
[groups, ~, group_keys] = findgroups(stats_table.Config, stats_table.CalibPercent);

% Extract values from group_keys
Config_list = splitapply(@(x) x(1), stats_table.Config, groups);
Pct_list    = splitapply(@(x) x(1), stats_table.CalibPercent, groups);

% Init output table
BestCalib_ByConfigPct = table();
BestCalib_ByConfigPct.Config       = Config_list;
BestCalib_ByConfigPct.CalibPercent = Pct_list;
BestCalib_ByConfigPct.BestType     = strings(size(Config_list));
BestCalib_ByConfigPct.BestMean     = zeros(size(Config_list));
BestCalib_ByConfigPct.BestStd      = zeros(size(Config_list));

% Select best CalibrationType per group
for g = 1:max(groups)
    rows = (groups == g);
    subset = stats_table(rows, :);

    [best_val, idx] = max(subset.Mean_Accuracy);
    BestCalib_ByConfigPct.BestType(g) = subset.CalibrationType(idx);
    BestCalib_ByConfigPct.BestMean(g) = best_val;
    BestCalib_ByConfigPct.BestStd(g)  = subset.Std_Accuracy(idx);
end


% -------------------------------------------------------------------------
% (Top/Worst 10) FEATURE EXTRACTION  into Summary File
% -------------------------------------------------------------------------

% Base path already defined as base_pipeline_path
folders = dir(fullfile(base_pipeline_path, '*samples*'));
folders = folders([folders.isdir]);

% Find unique sample sizes (e.g., 1000, 2000, etc.)
sample_sizes = [];
for i = 1:length(folders)
    tokens = regexp(folders(i).name, '^(\d+)samples', 'tokens');
    if ~isempty(tokens)
        sample_size = str2double(tokens{1}{1});
        if sample_size >= 1000  % Only real datasets, skip 50 samples etc
            sample_sizes = [sample_sizes; sample_size];
        end
    end
end
sample_sizes = unique(sample_sizes);

% Now for each UNIQUE sample size, extract once
AllTop = table();
AllWorst = table();
F_data = {};

for s = 1:length(sample_sizes)
    sample_size = sample_sizes(s);
    folder_pattern = sprintf('%dsamples*', sample_size);
    matching_folders = dir(fullfile(base_pipeline_path, folder_pattern));
    matching_folders = matching_folders([matching_folders.isdir]);

    if isempty(matching_folders)
        warning('[!] No folder found for %d samples', sample_size);
        continue;
    end

    % Pick the FIRST matching folder
    folder_name = matching_folders(1).name;
    fprintf('\n[INFO] Extracting Top/Worst features from: %s\n', folder_name);

    stats_file = dir(fullfile(base_pipeline_path, folder_name, 'PreCalib_*Stats.xlsx'));
    if isempty(stats_file)
        warning('[!] No stats file found in: %s', folder_name);
        continue;
    end

    stats_path = fullfile(stats_file(1).folder, stats_file(1).name);

    try
        T_top = readtable(stats_path, 'Sheet', 'Total_Top_Features');
        T_worst = readtable(stats_path, 'Sheet', 'Total_Worst_Features');

        T_top.SampleSet = repmat(sprintf('%dsamples', sample_size), height(T_top), 1);
        T_worst.SampleSet = repmat(sprintf('%dsamples', sample_size), height(T_worst), 1);

        T_top = sortrows(T_top, 'AvgMetricScore', 'descend');
        T_worst = sortrows(T_worst, 'AvgMetricScore', 'ascend');

        AllTop = [AllTop; T_top(1:min(10, height(T_top)), :)];
        AllWorst = [AllWorst; T_worst(1:min(10, height(T_worst)), :)];

        % Read all F_* sheets
        [~, sheets] = xlsfinfo(stats_path);
        F_sheets = sheets(contains(sheets, 'F_'));
        for s_idx = 1:length(F_sheets)
            try
                F_table = readtable(stats_path, 'Sheet', F_sheets{s_idx});
                F_data{end+1} = struct(...
                    'Folder', folder_name, ...
                    'SampleSet', sprintf('%dsamples', sample_size), ...
                    'Sheet', F_sheets{s_idx}, ...
                    'Table', F_table);
            catch ME
                warning('[!] Failed reading sheet %s in %s: %s', F_sheets{s_idx}, stats_path, ME.message);
            end
        end

    catch ME
        warning('[!] Failed to read %s: %s', stats_path, ME.message);
    end
end

% -------------------------------------------------------------------------
% EXPORT TO EXCEL
% -------------------------------------------------------------------------
AllWithin = sortrows(AllWithin, {'Dataset','TotalSamples','CalibSamples'});
summary_within = sortrows(summary_within, {'Dataset','TotalSamples'});
AllCross = sortrows(AllCross, {'Dataset','TotalSamples','Stage','CalibrationType'});
summary_cross = sortrows(summary_cross, {'Dataset','TotalSamples','Stage','CalibrationType'});

summary_file = 'v1_Full_Final_Summary.xlsx';                                        % !! CHANGE SUMMARY FILE VERSION IF NEW !!
writetable(AllWithin, summary_file, 'Sheet', 'Within_All');
writetable(summary_within_configs, summary_file, 'Sheet', 'Summary_Within_Configs');
writetable(summary_within_modeltypes, summary_file, 'Sheet', 'Summary_Within_Modeltypes');
writetable(summary_modeltypes_25wCsp, summary_file, ...
    'Sheet', 'Summary_Modeltypes_25wCsp');
writetable(modeltype_summary_25wCsp, summary_file, 'Sheet', 'MeanModelTypes_25wCsp');
writetable(summary_within, summary_file, 'Sheet', 'Summary_Within_All');
writetable(AllCross, summary_file, 'Sheet', 'Cross_All');
writetable(summary_cross, summary_file, 'Sheet', 'Summary_Cross_All');
writetable(summary_cross_configs, summary_file, 'Sheet', 'Summary_Cross_Configs');
writetable(summary_cross_modeltypes, summary_file, 'Sheet', 'Summary_Cross_Modeltypes');
writetable(BestCalib_ByDatasetPct, summary_file, 'Sheet', 'BestCalib_ByDatasetPct');
writetable(BestCalib_ByConfigPct, summary_file, 'Sheet', 'BestCalib_ByConfigPct');
writetable(AllDeltaSummary, summary_file, 'Sheet', 'Delta_All');
writetable(Summary_25wCsp, summary_file, 'Sheet', 'Delta_Summary_25wCsp');
writetable(Summary_Delta_All, summary_file, 'Sheet', 'Summary_Delta_All');
writetable(summary_delta_overall, summary_file, 'Sheet', 'Summary_Delta_Overall');
writetable(AllTop, summary_file, 'Sheet', 'Top10_Features');
writetable(AllWorst, summary_file, 'Sheet', 'Worst10_Features');


% -------------------------------------------------------------------------
% EXTEND Top10 and Worst10 Features with p-value and Cohen's d Statistics
% -------------------------------------------------------------------------

fprintf('\n[INFO] Extending Top10 and Worst10 Features with Mean/Min/Max p-value and Cohen''s d...\n');

% Re-load the Summary File (where Top10/Worst10 are already written)
summary_file = 'v1_Full_Final_Summary.xlsx';                                            % !! CHANGE SUMMARY FILE VERSION IF NEW !! 
top_table = readtable(summary_file, 'Sheet', 'Top10_Features');
worst_table = readtable(summary_file, 'Sheet', 'Worst10_Features');

% Process both Top10 and Worst10
top_table_ext = add_stats(top_table, F_data);
worst_table_ext = add_stats(worst_table, F_data);

writetable(top_table_ext, summary_file, 'Sheet', 'Top10_Features');
writetable(worst_table_ext, summary_file, 'Sheet', 'Worst10_Features');

% Update AllTop and AllWorst with stats as well
AllTop = add_stats(AllTop, F_data);
AllWorst = add_stats(AllWorst, F_data);


% -------------------------------------------------------------------------
% OVERALL RANKING: Aggregate Unique Top 10 and Worst 10 Features Across All Samples
% -------------------------------------------------------------------------

% Aggregate: compute both MEAN and MAX for Top, MEAN and MIN for Worst
TopAgg_mean = varfun(@mean, AllTop, 'InputVariables', 'AvgMetricScore', ...
                     'GroupingVariables', 'FeatureName');
TopAgg_max = varfun(@max, AllTop, 'InputVariables', 'AvgMetricScore', ...
                    'GroupingVariables', 'FeatureName');

WorstAgg_mean = varfun(@mean, AllWorst, 'InputVariables', 'AvgMetricScore', ...
                       'GroupingVariables', 'FeatureName');
WorstAgg_min = varfun(@min, AllWorst, 'InputVariables', 'AvgMetricScore', ...
                      'GroupingVariables', 'FeatureName');

% Join Mean and Max/Min
TopAgg = join(TopAgg_mean, TopAgg_max);
WorstAgg = join(WorstAgg_mean, WorstAgg_min);

% Rename columns for clarity
TopAgg.Properties.VariableNames{'mean_AvgMetricScore'} = 'Mean_AvgMetricScore';
TopAgg.Properties.VariableNames{'max_AvgMetricScore'} = 'Max_AvgMetricScore';
WorstAgg.Properties.VariableNames{'mean_AvgMetricScore'} = 'Mean_AvgMetricScore';
WorstAgg.Properties.VariableNames{'min_AvgMetricScore'} = 'Min_AvgMetricScore';

% --- Now sort by Mean value ---
TopAgg = sortrows(TopAgg, 'Mean_AvgMetricScore', 'descend');
WorstAgg = sortrows(WorstAgg, 'Mean_AvgMetricScore', 'ascend');

% Select Top 10 and Worst 10
TopOverall = TopAgg(1:min(10, height(TopAgg)), :);
WorstOverall = WorstAgg(1:min(10, height(WorstAgg)), :);

% --- Remove ClassTrend if it somehow appears ---
if ismember('ClassTrend', TopOverall.Properties.VariableNames)
    TopOverall = removevars(TopOverall, 'ClassTrend');
end
if ismember('ClassTrend', WorstOverall.Properties.VariableNames)
    WorstOverall = removevars(WorstOverall, 'ClassTrend');
end

% -------------------------------------------------------------------------
% ADD Mean/Min/Max p-value and Cohen's d to TopOverall and WorstOverall
% -------------------------------------------------------------------------

% Initialize columns
TopOverall.Mean_p = nan(height(TopOverall),1);
TopOverall.Min_p  = nan(height(TopOverall),1);
TopOverall.Max_p  = nan(height(TopOverall),1);
TopOverall.Mean_d = nan(height(TopOverall),1);
TopOverall.Min_d  = nan(height(TopOverall),1);
TopOverall.Max_d  = nan(height(TopOverall),1);

WorstOverall.Mean_p = nan(height(WorstOverall),1);
WorstOverall.Min_p  = nan(height(WorstOverall),1);
WorstOverall.Max_p  = nan(height(WorstOverall),1);
WorstOverall.Mean_d = nan(height(WorstOverall),1);
WorstOverall.Min_d  = nan(height(WorstOverall),1);
WorstOverall.Max_d  = nan(height(WorstOverall),1);

% Fill TopOverall
for i = 1:height(TopOverall)
    feat = TopOverall.FeatureName(i);
    rows = strcmp(AllTop.FeatureName, feat);

    if any(rows)
        p_vals = AllTop.Mean_p(rows);
        d_vals = AllTop.Mean_d(rows);

        p_vals = p_vals(~isnan(p_vals));
        d_vals = d_vals(~isnan(d_vals));

        if ~isempty(p_vals)
            TopOverall.Mean_p(i) = mean(p_vals);
            TopOverall.Min_p(i)  = min(p_vals);
            TopOverall.Max_p(i)  = max(p_vals);
        end

        if ~isempty(d_vals)
            TopOverall.Mean_d(i) = mean(d_vals);
            TopOverall.Min_d(i)  = min(d_vals);
            TopOverall.Max_d(i)  = max(d_vals);
        end
    end
end

% Fill WorstOverall
for i = 1:height(WorstOverall)
    feat = WorstOverall.FeatureName(i);
    rows = strcmp(AllWorst.FeatureName, feat);

    if any(rows)
        p_vals = AllWorst.Mean_p(rows);
        d_vals = AllWorst.Mean_d(rows);

        p_vals = p_vals(~isnan(p_vals));
        d_vals = d_vals(~isnan(d_vals));

        if ~isempty(p_vals)
            WorstOverall.Mean_p(i) = mean(p_vals);
            WorstOverall.Min_p(i)  = min(p_vals);
            WorstOverall.Max_p(i)  = max(p_vals);
        end
        if ~isempty(d_vals)
            WorstOverall.Mean_d(i) = mean(d_vals);
            WorstOverall.Min_d(i)  = min(d_vals);
            WorstOverall.Max_d(i)  = max(d_vals);
        end
    end
end

writetable(TopOverall, summary_file, 'Sheet', 'Top10_Overall');
writetable(WorstOverall, summary_file, 'Sheet', 'Worst10_Overall');




% -------------------------------------------------------------------------
% PLOTTING
% -------------------------------------------------------------------------

% CALIBRATION STRATEGY COMPARISON — Figure 1 (Δ + Accuracy + Win Count)
calib_types = {'adapted', 'finetuned', 'finetuned_adapted'};
type_labels = {'Adapted', 'Finetuned', 'Finetuned+Adapted'};

% 1. Mean Delta & STD from Delta Summary
delta_vals = zeros(1, length(calib_types));
delta_stds = zeros(1, length(calib_types));
for i = 1:length(calib_types)
    idx = strcmp(summary_delta_overall.CalibrationType, calib_types{i});
    delta_vals(i) = summary_delta_overall.Mean_Delta(idx);
    delta_stds(i) = summary_delta_overall.Std_Delta(idx);
end

% 2. Mean Post Accuracy & STD from BestCalib_ByDatasetPct
acc_vals = zeros(1, length(calib_types));
acc_stds = zeros(1, length(calib_types));
for i = 1:length(calib_types)
    idx = strcmp(BestCalib_ByDatasetPct.BestType, calib_types{i});
    acc_vals(i) = mean(BestCalib_ByDatasetPct.BestMean(idx), 'omitnan');
    acc_stds(i) = std(BestCalib_ByDatasetPct.BestMean(idx), 'omitnan');
end

% 3. Count Wins
win_counts = zeros(1, numel(calib_types));
for i = 1:numel(calib_types)
    win_counts(i) = sum(BestCalib_ByDatasetPct.BestType == calib_types{i});
end

% Plot: All 3 panels side by side
figure('Color','w', 'Position', [100 100 1280 400]);

% --- Δ Accuracy (Improvement)
subplot(1,3,1);
bar(delta_vals, 0.6); hold on;
errorbar(1:length(delta_vals), delta_vals, delta_stds, 'k.', 'LineWidth', 1.5);
title('Mean Accuracy Gain (Δ)', 'FontWeight','bold');
ylabel('Δ Accuracy (%)');
set(gca, 'XTick', 1:length(type_labels), 'XTickLabel', type_labels, 'FontSize', 12);
ylim([0, max(delta_vals + delta_stds) + 1]);
grid on;

% Final Accuracy (Best-Performer Mean)
subplot(1,3,2);
bar(acc_vals, 0.6); hold on;
errorbar(1:length(acc_vals), acc_vals, acc_stds, 'k.', 'LineWidth', 1.5);
title('Mean Accuracy (Best Strategy Only)', 'FontWeight','bold');
ylabel('Accuracy (%)');
set(gca, 'XTick', 1:length(type_labels), 'XTickLabel', type_labels, 'FontSize', 12);
ylim([min(acc_vals - acc_stds) - 1, max(acc_vals + acc_stds) + 1]);
grid on;

% Win Count
subplot(1,3,3);
bar(win_counts);
set(gca, 'XTickLabel', type_labels, 'FontSize', 12);
ylabel('# Times Best');
title('Calibration Type Wins');
ylim([0, max(win_counts) + 1]);
grid on;

sgtitle('Comparison of Calibration Types — Δ vs Accuracy vs Frequency (Best Only | Delta_All)', 'FontSize', 14, 'FontWeight', 'bold');


% -------------------------------------------------------------------------
% Figure 2: Post Accuracy (Across All Cross-Results)
% -------------------------------------------------------------------------
T_post = AllCross(strcmp(AllCross.Stage, "Post"), :);
valid = ismember(T_post.CalibrationType, calib_types);
T_post = T_post(valid, :);

means = grpstats(T_post.ACCURACY, T_post.CalibrationType, 'mean');
stds  = grpstats(T_post.ACCURACY, T_post.CalibrationType, 'std');

% Get in same order
[~, order] = ismember(calib_types, unique(T_post.CalibrationType));
means = means(order);
stds = stds(order);

figure('Color','w', 'Position', [200 150 600 400]);
bar(means, 0.6); hold on;
errorbar(1:length(means), means, stds, 'k.', 'LineWidth', 1.5);
set(gca, 'XTick', 1:length(type_labels), 'XTickLabel', type_labels, 'FontSize', 12);
ylabel('Mean Accuracy (%)');
title('Post-Calibration Accuracy (All Cross Results | Summary_Cross_All)', 'FontWeight', 'bold');
ylim([floor(min(means - stds)) - 1, ceil(max(means + stds)) + 1]);
grid on;


% % -------------------------------------------------------------------------
% % COEFFICIENT OF VARIATION (CoV) ANALYSIS FOR CALIBRATION TYPES
% % -------------------------------------------------------------------------
% % Calibration types to analyze
% calib_types = {'adapted', 'finetuned', 'finetuned_adapted'};
% type_labels = {'Adapted', 'Finetuned', 'Finetuned+Adapted'};
% 
% % Select Post-Calibration Results
% T_post = AllCross(strcmp(AllCross.Stage, "Post"), :);
% valid_rows = ismember(T_post.CalibrationType, calib_types);
% T_post = T_post(valid_rows, :);
% 
% % Initialize
% mean_acc = zeros(1, numel(calib_types));
% std_acc = zeros(1, numel(calib_types));
% cov_acc = zeros(1, numel(calib_types));
% 
% % Compute Mean, STD, and CoV
% for i = 1:numel(calib_types)
%     calib = calib_types{i};
%     rows = strcmp(T_post.CalibrationType, calib);
% 
%     mean_acc(i) = mean(T_post.ACCURACY(rows), 'omitnan');
%     std_acc(i)  = std(T_post.ACCURACY(rows), 'omitnan');
%     cov_acc(i)  = std_acc(i) / mean_acc(i);  % Coefficient of Variation
% end
% 
% 
% % PLOT: Coefficient of Variation
% figure('Position', [300, 300, 700, 400], 'Color', 'w');
% bar(cov_acc, 0.5); hold on;
% ylabel('Coefficient of Variation (CoV)', 'FontSize', 12);
% set(gca, 'XTick', 1:length(type_labels), 'XTickLabel', type_labels, 'FontSize', 12);
% title('Relative Stability of Calibration Strategies (CoV) - Lower CoV = Higher Stability Across Runs', 'FontSize', 14, 'FontWeight', 'bold');
% grid on;
% ylim([0, max(cov_acc) + 0.05]);
% 
% % Annotate CoV values on bars
% for i = 1:length(cov_acc)
%     text(i, cov_acc(i) + 0.01, sprintf('%.2f', cov_acc(i)), ...
%         'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
% end


fprintf('\n✅ Saved Full Version specific Summary to: %s\n', summary_file);




% Function to compute stats for a given feature list
function T_extended = add_stats(T_in, F_data)
    num_features = height(T_in);

    Mean_p = nan(num_features,1);
    Min_p  = nan(num_features,1);
    Max_p  = nan(num_features,1);
    Mean_d = nan(num_features,1);
    Min_d  = nan(num_features,1);
    Max_d  = nan(num_features,1);

    for f = 1:num_features
        feature_name = string(T_in.FeatureName(f));
        sample_set = string(T_in.SampleSet(f));  % Use SampleSet matching!

        % Collect all p-values and cohens'd across matching F_data entries
        p_vals = [];
        d_vals = [];

        for j = 1:length(F_data)
            F_entry = F_data{j};

            % Match both FeatureName AND SampleSet (VERY IMPORTANT)
            if contains(F_entry.Folder, sample_set)
                F_table = F_entry.Table;
                match_idx = strcmp(string(F_table.FeatureName), feature_name);

                if any(match_idx)
                    if ismember('P_value', F_table.Properties.VariableNames)
                        p_vals = [p_vals; F_table.P_value(match_idx)];
                    end
                    if ismember('Cohens_d', F_table.Properties.VariableNames)
                        d_vals = [d_vals; F_table.Cohens_d(match_idx)];
                    end
                end
            end
        end

        % Clean NaNs
        p_vals = p_vals(~isnan(p_vals));
        d_vals = d_vals(~isnan(d_vals));

        % Compute Stats if any data
        if ~isempty(p_vals)
            Mean_p(f) = mean(p_vals);
            Min_p(f) = min(p_vals);
            Max_p(f) = max(p_vals);
        end
        if ~isempty(d_vals)
            Mean_d(f) = mean(d_vals);
            Min_d(f) = min(d_vals);
            Max_d(f) = max(d_vals);
        end
    end

    % Add Columns
    T_extended = T_in;
    T_extended.Mean_p = Mean_p;
    T_extended.Min_p = Min_p;
    T_extended.Max_p = Max_p;
    T_extended.Mean_d = Mean_d;
    T_extended.Min_d = Min_d;
    T_extended.Max_d = Max_d;
end

