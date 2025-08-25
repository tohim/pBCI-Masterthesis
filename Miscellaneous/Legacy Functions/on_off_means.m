%% OFFLINE AND ONLINE FEATURE MEANS


%% Offline 


features = {... % your 31 feature names
    'Theta Power', 'Alpha Power', 'Beta Power', 'Theta Alpha Ratio', 'Theta Beta Ratio', 'Engagement Index', ...
    'Theta Frontal', 'Theta Temporal', 'Theta Parietal', 'Theta Occipital', 'Alpha Frontal', 'Alpha Temporal', ...
    'Alpha Parietal', 'Alpha Occipital', 'Beta Frontal', 'Beta Temporal', 'Beta Parietal', 'Avg Coherence', ...
    'Theta Coherence', 'Alpha Coherence', 'Avg Mobility', 'Avg Complexity', 'Avg Entropy', 'Theta Entropy', ...
    'Alpha Entropy', 'CSP1_Low_Workload', 'CSP2_Low_Workload', 'CSP3_Low_Workload', ...
    'CSP1_High_Workload', 'CSP2_High_Workload', 'CSP3_High_Workload'};

% Combine all into one array
datasets = {'STEW', 'MATB Easy Diff', 'MATB Easy MedDiff', 'HEAT'};
colors = [30, 130, 255; 255, 160, 40; 180, 70, 200] / 255;  % blue, orange, purple


% STEW

%stew_mean = [2.399889097, 2.006476274, 1.266082705, 0.796413426, 1.825583638, 0.127701624, 2.642826673, 2.400558616, 2.17512304, 2.254375114, ...
%     2.061200488, 1.991421914, 1.905581956, 2.075708107, 1.290476126, 1.286451719, 1.226887259, 0.585908831, 0.662423891, 0.625240133, 0.391943186, ...
%     0.684079067, 3.561933142, 1.500283936, 1.120328686, -3.498327618, -3.482650225, -3.191521811, -1.295411637, -1.330118877, -1.326925086];
stew_features = load('1000_25wCsp_4sec_proc5_STEW_train_features.mat');
all_epoch_features_STEW = stew_features.train_features;

% MATB EASY DIFF 

%matb_easy_diff = [1.487255553, 0.956937575, 0.561089167, 0.765642691, 1.888674708, 0.125081522, 1.641641586, 1.354240094, 1.408030039, 1.585395264, ...
%     1.003376165, 0.871015517, 1.014647987, 1.199230397, 0.587838106, 0.512001812, 0.673841286, 0.577215433, 0.628459731, 0.587537168, 0.406173511, ...
%     0.751704007, 3.645646894, 1.70230516, 1.025987728, -2.295790329, -2.012560478, -2.056760942, -1.746009897, -1.793200482, -1.853734279];
matb_easy_diff_features = load('1000_25wCsp_4sec_proc5_MATB_easy_diff_train_features.mat');
all_epoch_features_MATB_easy_diff = matb_easy_diff_features.train_features;


% MATB EASY MED DIFF

%matb_easy_meddiff = [1.464663355, 0.947016185, 0.549194963, 0.759068085, 1.874011242, 0.126212524, 1.607511387, 1.330234288, 1.390724404, 1.567373253, ...
%     0.991635807, 0.854874575, 1.005489888, 1.172840515, 0.572979064, 0.498785338, 0.651755605, 0.576538575, 0.628793629, 0.587157043, 0.408312879, ...
%     0.753414312, 3.654815708, 1.716598739, 1.029757458, -2.26416521, -2.032664096, -2.05964425, -1.800078402, -1.815495658, -1.839588378];
matb_easy_med_diff_features = load('1000_25wCsp_4sec_proc5_MATB_easy_meddiff_train_features.mat');
all_epoch_features_MATB_easy_meddiff = matb_easy_med_diff_features.train_features;


% HEAT

%heat = [3.906467988, 2.978227663, 2.273837482, 0.957777285, 2.031124219, 0.117923222, 4.00119885, 3.912789358, 3.748088602, 3.852960798, ...
%     3.026821506, 2.957785499, 2.906658515, 2.971426696, 2.270771042, 2.263120455, 2.433119825, 0.695676455, 0.732406935, 0.690018839, 0.348474914, ...
%     0.727267429, 3.455483842, 1.502801844, 0.777900403, -1.432027753, -1.664114964, -2.078513185, -2.014931179, -2.384903631, -2.94736859];
heat_features = load('1000_25wCsp_4sec_proc5_HEATCHAIR_train_features.mat');
all_epoch_features_HEAT = heat_features.train_features;

%% Plots
features_per_fig = 8;  % fewer plots per figure = larger plots
n_features = numel(features);
n_figs = ceil(n_features / features_per_fig);

for fig_idx = 1:n_figs
    figure('Color','w', 'Position', [100 100 800 1000]);  % wider and taller

    % Use vertical layout: one column, multiple rows
    t = tiledlayout(features_per_fig, 1, ...
        'TileSpacing', 'compact', ...
        'Padding', 'compact');

    start_idx = (fig_idx-1)*features_per_fig + 1;
    end_idx = min(fig_idx*features_per_fig, n_features);
    num_subplots = end_idx - start_idx + 1;

    for i = 1:num_subplots
        feature_idx = start_idx + i - 1;

        stew_vals  = all_epoch_features_STEW(:, feature_idx);
        matb1_vals = all_epoch_features_MATB_easy_diff(:, feature_idx);
        matb2_vals = all_epoch_features_MATB_easy_meddiff(:, feature_idx);
        heat_vals  = all_epoch_features_HEAT(:, feature_idx);

        nexttile;

        boxplot([stew_vals; matb1_vals; matb2_vals; heat_vals], ...
                [repmat({'STEW'}, length(stew_vals), 1); ...
                 repmat({'MATB Easy Diff'}, length(matb1_vals), 1); ...
                 repmat({'MATB Easy MedDiff'}, length(matb2_vals), 1); ...
                 repmat({'HEAT'}, length(heat_vals), 1)], ...
                 'Symbol', '.', 'Widths', 0.6);

        title(features{feature_idx}, 'FontWeight','bold', 'Interpreter','none');
        set(gca, 'FontSize', 10);
        grid on;
    end

    sgtitle(sprintf('Feature Distributions per Dataset (%d–%d)', start_idx, end_idx), ...
            'FontWeight','bold', 'FontSize', 14);
end


%% Global min max

% Combine all data
combined_values = [
    all_epoch_features_STEW(:);
    all_epoch_features_MATB_easy_diff(:);
    all_epoch_features_MATB_easy_meddiff(:);
    all_epoch_features_HEAT(:)
];

global_min = min(combined_values);
global_max = max(combined_values);
value_range = global_max - global_min;
global_mean = mean(combined_values);

fprintf('Global mean across all features and datasets: %.3f\n', global_mean);
fprintf('Global min: %.3f | Global max: %.3f | Range: %.3f\n', ...
        global_min, global_max, value_range);

% with 95% confidence interval 

% Combine all feature values across datasets
combined_values = [
    all_epoch_features_STEW(:);
    all_epoch_features_MATB_easy_diff(:);
    all_epoch_features_MATB_easy_meddiff(:);
    all_epoch_features_HEAT(:)
];

% Compute statistics
global_mean = mean(combined_values);
global_std = std(combined_values);
n = numel(combined_values);

% 95% confidence interval
ci_range = 1.96 * (global_std / sqrt(n));
ci_low = global_mean - ci_range;
ci_high = global_mean + ci_range;

% Display
fprintf('Global STD: %.3f\n', global_std);
fprintf('Global mean: %.3f\n', global_mean);
fprintf('95%% Confidence Interval: [%.3f, %.3f]\n', ci_low, ci_high);
fprintf('Global value range: [%.3f, %.3f]\n', min(combined_values), max(combined_values));



%% Generate table

% Prepare dataset names and feature matrix
dataset_names = {'STEW', 'MATB Easy Diff', 'MATB Easy MedDiff', 'HEAT'};
all_datasets = {
    all_epoch_features_STEW;
    all_epoch_features_MATB_easy_diff;
    all_epoch_features_MATB_easy_meddiff;
    all_epoch_features_HEAT
};

n_features = numel(features);
n_datasets = numel(dataset_names);

% Preallocate cell arrays
Feature = cell(n_features * n_datasets, 1);
Dataset = cell(n_features * n_datasets, 1);
Mean = zeros(n_features * n_datasets, 1);
StdDev = zeros(n_features * n_datasets, 1);
MinVal = zeros(n_features * n_datasets, 1);
MaxVal = zeros(n_features * n_datasets, 1);

% Compute statistics: loop over features first
row = 1;
for f = 1:n_features
    for d = 1:n_datasets
        data = all_datasets{d}(:, f);
        Feature{row} = features{f};
        Dataset{row} = dataset_names{d};
        Mean(row)    = mean(data);
        StdDev(row)  = std(data);
        MinVal(row)  = min(data);
        MaxVal(row)  = max(data);
        row = row + 1;
    end
end

% Create final table
stats_table = table(Feature, Dataset, Mean, StdDev, MinVal, MaxVal);

% Optional: display first rows
disp(stats_table(1:12,:));

% Optional: export
writetable(stats_table, 'feature_stats_per_dataset.txt', 'Delimiter', '\t');

%%

% Define feature and dataset names
features = {... 
    'Theta Power', 'Alpha Power', 'Beta Power', 'Theta Alpha Ratio', 'Theta Beta Ratio', 'Engagement Index', ...
    'Theta Frontal', 'Theta Temporal', 'Theta Parietal', 'Theta Occipital', 'Alpha Frontal', 'Alpha Temporal', ...
    'Alpha Parietal', 'Alpha Occipital', 'Beta Frontal', 'Beta Temporal', 'Beta Parietal', 'Avg Coherence', ...
    'Theta Coherence', 'Alpha Coherence', 'Avg Mobility', 'Avg Complexity', 'Avg Entropy', 'Theta Entropy', ...
    'Alpha Entropy', 'CSP1_Low_Workload', 'CSP2_Low_Workload', 'CSP3_Low_Workload', ...
    'CSP1_High_Workload', 'CSP2_High_Workload', 'CSP3_High_Workload'};

datasets = {'STEW', 'MATB Easy Diff', 'MATB Easy MedDiff', 'HEAT'};

% Load feature matrices (rows: epochs, columns: features)
load('1000_25wCsp_4sec_proc5_STEW_train_features.mat'); stew = train_features;
load('1000_25wCsp_4sec_proc5_MATB_easy_diff_train_features.mat'); matb1 = train_features;
load('1000_25wCsp_4sec_proc5_MATB_easy_meddiff_train_features.mat'); matb2 = train_features;
load('1000_25wCsp_4sec_proc5_HEATCHAIR_train_features.mat'); heat = train_features;

all_data = {stew, matb1, matb2, heat};

% Preallocate results
out_feature = {};
out_dataset = {};
out_mean = [];
out_std = [];
out_min = [];
out_max = [];

% Construct table row-by-row, grouped by feature
for f = 1:numel(features)
    for d = 1:numel(datasets)
        vals = all_data{d}(:, f);
        out_feature{end+1,1} = features{f};
        out_dataset{end+1,1} = datasets{d};
        out_mean(end+1,1) = mean(vals);
        out_std(end+1,1) = std(vals);
        out_min(end+1,1) = min(vals);
        out_max(end+1,1) = max(vals);
    end
end

% Build MATLAB table
T = table(out_feature, out_dataset, out_mean, out_std, out_min, out_max, ...
    'VariableNames', {'Feature', 'Dataset', 'Mean', 'StdDev', 'MinVal', 'MaxVal'});

% Display table in Command Window
disp(T);

% Export to LaTeX
latex_code = sprintf(['\\begin{tabular}{llrrrr}\n\\toprule\nFeature & Dataset & Mean & StdDev & Min & Max \\\\\n\\midrule\n']);
for i = 1:height(T)
    latex_code = [latex_code sprintf('%s & %s & %.4f & %.4f & %.4f & %.4f \\\\\n', ...
        T.Feature{i}, T.Dataset{i}, T.Mean(i), T.StdDev(i), T.MinVal(i), T.MaxVal(i))];
end
latex_code = [latex_code '\bottomrule\n\end{tabular}'];

% Save to .tex file
fid = fopen('feature_stats_table.tex', 'w');
fprintf(fid, '%s', latex_code);
fclose(fid);



