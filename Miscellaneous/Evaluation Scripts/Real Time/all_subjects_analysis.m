%% All Subjects Analysis

%% Collecting all data

all_data_calib = [];
all_data_exp = [];

subject_data = struct;
subject_calib_models = struct;

% === Subject selection ===
kept_subjects = [1 2 3 4 6 7 8 9 10];   % removed: 5 and 11
n_subjects    = numel(kept_subjects);

for i = 1:n_subjects % go from 1 to amount of subjects to analyse

    subject_number = kept_subjects(i);  % actual file index, e.g., 10

    disp(subject_number);

    sub_calib = load(sprintf('Subject%d_CalibrationLog.mat', subject_number));
    sub_exp   = load(sprintf('Subject%d_Results.mat', subject_number));

    subject_calib_models(i).sub_calib_stew = load(sprintf('Subject%d_CalibratedModel_STEW.mat', subject_number));
    subject_calib_models(i).sub_calib_heat = load(sprintf('Subject%d_CalibratedModel_HEAT.mat', subject_number));
    subject_calib_models(i).sub_calib_matb = load(sprintf('Subject%d_CalibratedModel_MATB.mat', subject_number));

    max_calib = length(sub_calib.calibration_log);
    min_calib = max_calib - 600 + 1;

    subject_data.calib = sub_calib.calibration_log(min_calib:max_calib);
    subject_data.exp   = sub_exp.experiment_log(1:300);

    all_data_calib = [all_data_calib, subject_data.calib];
    all_data_exp   = [all_data_exp, subject_data.exp];

end

% Check for availability of all data
required_fields = {'features_STEW', 'features_HEAT', 'features_MATB'};
if all(isfield(subject_data.calib, required_fields))
    % proceed
else
    warning('Missing features in Subject %d', subject_number);
end

% Add block count to the all_data_exp
epochs_per_subject = 300;
epochs_per_block = 30;
total_epochs = numel(all_data_exp);

block_numbers = zeros(total_epochs, 1);

for i = 1:total_epochs
    subj_idx = ceil(i / epochs_per_subject);                % which sub is this epoch from?
    subj_epoch_idx = mod(i-1, epochs_per_subject) + 1;      % Epoch index within the subject
    block_idx = ceil(subj_epoch_idx / epochs_per_block);    % block number within the subject
    block_numbers(i) = block_idx;
end

% Use a loop instead:
for i = 1:total_epochs
    all_data_exp(i).block = block_numbers(i);
end



%% MEAN DIFFERENCES ALL CALIB

% CALIBRATION DATA

true_label = cell2mat({all_data_calib.true_label})';

low_idx = true_label == 0;
high_idx = true_label == 1;

% Convert features from cell to matrix
all_epoch_features_STEW = cell2mat({all_data_calib.features_STEW}');
all_epoch_features_HEAT = cell2mat({all_data_calib.features_HEAT}');
all_epoch_features_MATB = cell2mat({all_data_calib.features_MATB}');

base_low = all_epoch_features_STEW(low_idx, 1:25);
base_high = all_epoch_features_STEW(high_idx, 1:25);

stew_low = all_epoch_features_STEW(low_idx, 26:31);
stew_high = all_epoch_features_STEW(high_idx, 26:31);

heat_low = all_epoch_features_HEAT(low_idx, 26:31);
heat_high = all_epoch_features_HEAT(high_idx, 26:31);

matb_low = all_epoch_features_MATB(low_idx, 26:31);
matb_high = all_epoch_features_MATB(high_idx, 26:31);

combined_low_calib = [base_low, stew_low, heat_low, matb_low];
combined_high_calib = [base_high, stew_high, heat_high, matb_high];

mean_low = mean(combined_low_calib, 1);
mean_high = mean(combined_high_calib, 1);


% EXPERIMENT REORDERED
features = {... % add all 31 names
    'Theta Power', 'Alpha Power', 'Beta Power', ...
    'Theta Alpha Ratio', 'Theta Beta Ratio', 'Engagement Index', ...
    'Theta Frontal', 'Theta Temporal', 'Theta Parietal', 'Theta Occipital', ...
    'Alpha Frontal', 'Alpha Temporal', 'Alpha Parietal', 'Alpha Occipital', ...
    'Beta Frontal', 'Beta Temporal', 'Beta Parietal', ...
    'Avg Coherence', 'Theta Coherence', 'Alpha Coherence', ...
    'Avg Mobility', 'Avg Complexity', 'Avg Entropy', ...
    'Theta Entropy', 'Alpha Entropy', ...
    'CSP1 LOW STEW', 'CSP2 LOW STEW', 'CSP3 LOW STEW', ...
    'CSP1 HIGH STEW', 'CSP2 HIGH STEW', 'CSP3 HIGH STEW', ...
    'CSP1 LOW HEAT', 'CSP2 LOW HEAT', 'CSP3 LOW HEAT', ...
    'CSP1 HIGH HEAT', 'CSP2 HIGH HEAT', 'CSP3 HIGH HEAT', ...
    'CSP1 LOW MATB', 'CSP2 LOW MATB', 'CSP3 LOW MATB', ...
    'CSP1 HIGH MATB', 'CSP2 HIGH MATB', 'CSP3 HIGH MATB'};


% Compute delta
delta_mean_calib = mean_high - mean_low;

% Sort for plotting
[delta_sorted, idx] = sort(delta_mean_calib);
features_sorted = features(idx);

% Assign colors based on trend
bar_colors = repmat([0.2 0.7 0.3], length(delta_sorted), 1);  % green default
bar_colors(delta_sorted < 0, :) = repmat([0.8 0.2 0.2], sum(delta_sorted < 0), 1);  % red for negative

% Plot only once
figure('Color','w', 'Position', [100 100 1000 800]);
b = barh(delta_sorted, 'FaceColor','flat', 'EdgeColor','none');  % Single call
b.CData = bar_colors;

% Y labels and formatting
yticks(1:length(features_sorted));
yticklabels(features_sorted);
set(gca, 'YDir', 'reverse');  % Top = highest bar
xlabel('\Delta Mean (High - Low)', 'FontWeight', 'bold');
title('ALL CALIBRATION DATA Mean Differences Between HIGH and LOW MWL', 'FontWeight', 'bold', 'FontSize', 14);
line([0 0], ylim, 'Color', [0.2 0.2 0.2], 'LineStyle', '--');

% Font and aesthetics
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'TickLabelInterpreter', 'none');
box off;
grid on;



%% MEAN DIFFERENCES ALL EXPERIMENT

% Only look at predictions vs true labels before applying "ADAPT"
pred_idxs = isnan([all_data_exp.adapted_epochs]);

% Convert features from cell to matrix
all_epoch_features_STEW = cell2mat({all_data_exp(pred_idxs).STEW_features}');
all_epoch_features_HEAT = cell2mat({all_data_exp(pred_idxs).HEAT_features}');
all_epoch_features_MATB = cell2mat({all_data_exp(pred_idxs).MATB_features}');

true_label = cell2mat({all_data_exp(pred_idxs).true_label})';

low_idx = true_label == 0;
high_idx = true_label == 1;

base_low = all_epoch_features_STEW(low_idx, 1:25);
base_high = all_epoch_features_STEW(high_idx, 1:25);

stew_low = all_epoch_features_STEW(low_idx, 26:31);
stew_high = all_epoch_features_STEW(high_idx, 26:31);

heat_low = all_epoch_features_HEAT(low_idx, 26:31);
heat_high = all_epoch_features_HEAT(high_idx, 26:31);

matb_low = all_epoch_features_MATB(low_idx, 26:31);
matb_high = all_epoch_features_MATB(high_idx, 26:31);

combined_low_exp = [base_low, stew_low, heat_low, matb_low];
combined_high_exp = [base_high, stew_high, heat_high, matb_high];

mean_low = mean(combined_low_exp, 1);
mean_high = mean(combined_high_exp, 1);


% EXPERIMENT REORDERED
features = {... % add all 31 names
    'Theta Power', 'Alpha Power', 'Beta Power', ...
    'Theta Alpha Ratio', 'Theta Beta Ratio', 'Engagement Index', ...
    'Theta Frontal', 'Theta Temporal', 'Theta Parietal', 'Theta Occipital', ...
    'Alpha Frontal', 'Alpha Temporal', 'Alpha Parietal', 'Alpha Occipital', ...
    'Beta Frontal', 'Beta Temporal', 'Beta Parietal', ...
    'Avg Coherence', 'Theta Coherence', 'Alpha Coherence', ...
    'Avg Mobility', 'Avg Complexity', 'Avg Entropy', ...
    'Theta Entropy', 'Alpha Entropy', ...
    'CSP1 LOW STEW', 'CSP2 LOW STEW', 'CSP3 LOW STEW', ...
    'CSP1 HIGH STEW', 'CSP2 HIGH STEW', 'CSP3 HIGH STEW', ...
    'CSP1 LOW HEAT', 'CSP2 LOW HEAT', 'CSP3 LOW HEAT', ...
    'CSP1 HIGH HEAT', 'CSP2 HIGH HEAT', 'CSP3 HIGH HEAT', ...
    'CSP1 LOW MATB', 'CSP2 LOW MATB', 'CSP3 LOW MATB', ...
    'CSP1 HIGH MATB', 'CSP2 HIGH MATB', 'CSP3 HIGH MATB'};


% Compute delta
delta_mean_exp = mean_high - mean_low;

% Sort for plotting
[delta_sorted, idx] = sort(delta_mean_exp);
features_sorted = features(idx);

% Assign colors based on trend
bar_colors = repmat([0.2 0.7 0.3], length(delta_sorted), 1);  % green default
bar_colors(delta_sorted < 0, :) = repmat([0.8 0.2 0.2], sum(delta_sorted < 0), 1);  % red for negative

% Plot only once
figure('Color','w', 'Position', [100 100 1000 800]);
b = barh(delta_sorted, 'FaceColor','flat', 'EdgeColor','none');  % Single call
b.CData = bar_colors;

% Y labels and formatting
yticks(1:length(features_sorted));
yticklabels(features_sorted);
set(gca, 'YDir', 'reverse');  % Top = highest bar
xlabel('\Delta Mean (High - Low)', 'FontWeight', 'bold');
title('ALL EXPERIMENT DATA Mean Differences Between HIGH and LOW MWL', 'FontWeight', 'bold', 'FontSize', 14);
line([0 0], ylim, 'Color', [0.2 0.2 0.2], 'LineStyle', '--');

% Font and aesthetics
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'TickLabelInterpreter', 'none');
box off;
grid on;


%% OVERLAY MEAN DIFFERENCES OF CALIB AND EXP

% Assume:
% delta_mean_calib  = mean_high_calib - mean_low_calib;
% delta_mean_exp    = mean_high_exp - mean_low_exp;
% features          = { ... your feature list ... };

features = {... % add all 31 names
    'Theta Power', 'Alpha Power', 'Beta Power', ...
    'Theta Alpha Ratio', 'Theta Beta Ratio', 'Engagement Index', ...
    'Theta Frontal', 'Theta Temporal', 'Theta Parietal', 'Theta Occipital', ...
    'Alpha Frontal', 'Alpha Temporal', 'Alpha Parietal', 'Alpha Occipital', ...
    'Beta Frontal', 'Beta Temporal', 'Beta Parietal', ...
    'Avg Coherence', 'Theta Coherence', 'Alpha Coherence', ...
    'Avg Mobility', 'Avg Complexity', 'Avg Entropy', ...
    'Theta Entropy', 'Alpha Entropy', ...
    'CSP1 LOW STEW', 'CSP2 LOW STEW', 'CSP3 LOW STEW', ...
    'CSP1 HIGH STEW', 'CSP2 HIGH STEW', 'CSP3 HIGH STEW', ...
    'CSP1 LOW HEAT', 'CSP2 LOW HEAT', 'CSP3 LOW HEAT', ...
    'CSP1 HIGH HEAT', 'CSP2 HIGH HEAT', 'CSP3 HIGH HEAT', ...
    'CSP1 LOW MATB', 'CSP2 LOW MATB', 'CSP3 LOW MATB', ...
    'CSP1 HIGH MATB', 'CSP2 HIGH MATB', 'CSP3 HIGH MATB'};

% 1. Sort features by calibration delta
[delta_sorted, idx] = sort(delta_mean_exp);
features_sorted = features(idx);
calib_sorted = delta_mean_calib(idx);

% 2. Combine for grouped horizontal bar plot
data = [delta_sorted(:), calib_sorted(:)];

% 3. Set up plot
figure('Color','w', 'Position', [100 100 1200 800]);
b = barh(data, 'grouped');
hold on;

% 4. Add vertical zero line
xline(0, '--', 'Color', [0.3 0.3 0.3]);

% 5. Assign colors
b(2).FaceColor = [0.7 0.2 0.2];  % red for calibration
b(1).FaceColor = [0.2 0.2 0.7];  % blue for experiment

% 6. Labels and formatting
yticks(1:length(features_sorted));
yticklabels(features_sorted);
set(gca, 'YDir', 'reverse');  % Top = highest feature
xlabel('\Delta Mean (High - Low)', 'FontWeight', 'bold', 'FontSize', 13);
legend({'Experiment', 'Calibration'}, 'Location', 'southoutside', ...
    'Orientation', 'horizontal', 'FontSize', 12);

title('Feature-wise \Delta Mean: Calibration vs. Experiment (sorted by Experiment)', ...
    'FontWeight', 'bold', 'FontSize', 16);

% 7. Aesthetics
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);
box off;
grid on;


%% Descriptive statistics for CALIBRATION and EXPERIMENT phases (separately)

% Ensure you have 'features' (1x43 cell array of names)
% and the four matrices:
% combined_low_calib, combined_high_calib, combined_low_exp, combined_high_exp
% Each: [nEpochs x 43]

% Combine LOW & HIGH within each phase
calib_all = [combined_low_calib; combined_high_calib];
exp_all   = [combined_low_exp;  combined_high_exp];

% Compute and save per-feature descriptive stats for each phase
calib_desc = compute_phase_descriptives(calib_all, features, 'calibration_descriptive_stats.csv', 'Calibration');
exp_desc   = compute_phase_descriptives(exp_all,   features, 'experiment_descriptive_stats.csv',  'Experiment');




%% GLOBAL statistics and normality, t-test/ mann-whitney-u and effect size

% Feature names (43 entries)
features = {... % add all 31 names
    'Theta Power', 'Alpha Power', 'Beta Power', ...
    'Theta Alpha Ratio', 'Theta Beta Ratio', 'Engagement Index', ...
    'Theta Frontal', 'Theta Temporal', 'Theta Parietal', 'Theta Occipital', ...
    'Alpha Frontal', 'Alpha Temporal', 'Alpha Parietal', 'Alpha Occipital', ...
    'Beta Frontal', 'Beta Temporal', 'Beta Parietal', ...
    'Avg Coherence', 'Theta Coherence', 'Alpha Coherence', ...
    'Avg Mobility', 'Avg Complexity', 'Avg Entropy', ...
    'Theta Entropy', 'Alpha Entropy', ...
    'CSP1 LOW STEW', 'CSP2 LOW STEW', 'CSP3 LOW STEW', ...
    'CSP1 HIGH STEW', 'CSP2 HIGH STEW', 'CSP3 HIGH STEW', ...
    'CSP1 LOW HEAT', 'CSP2 LOW HEAT', 'CSP3 LOW HEAT', ...
    'CSP1 HIGH HEAT', 'CSP2 HIGH HEAT', 'CSP3 HIGH HEAT', ...
    'CSP1 LOW MATB', 'CSP2 LOW MATB', 'CSP3 LOW MATB', ...
    'CSP1 HIGH MATB', 'CSP2 HIGH MATB', 'CSP3 HIGH MATB'};

% Run for calibration
calib_stats = analyze_phase(combined_low_calib, combined_high_calib, features);

% Run for experiment
exp_stats = analyze_phase(combined_low_exp, combined_high_exp, features);

% Save to CSV for later
writetable(calib_stats, 'calibration_feature_stats.csv');
writetable(exp_stats, 'experiment_feature_stats.csv');

% Show summary
% disp('Calibration stats:');
% disp(calib_stats);
%
% disp('Experiment stats:');
% disp(exp_stats);


%% =========================
%  EXTENDED FEATURE ANALYSIS
%  =========================
% Requires:
%   features (1x43 cellstr)
%   combined_low_calib, combined_high_calib, combined_low_exp, combined_high_exp
% Optional (enables subject-level analyses):
%   all_data_calib (struct array with fields: true_label, features_STEW, features_HEAT, features_MATB)
%   all_data_exp   (struct array with fields: true_label, STEW_features, HEAT_features, MATB_features, adapted_epochs)
%   n_subjects (scalar), epochs_per_subject (calib/exp) if not inferable

assert(exist('features','var')==1, 'Missing "features".');
assert(all([exist('combined_low_calib','var'), exist('combined_high_calib','var'), ...
    exist('combined_low_exp','var'),   exist('combined_high_exp','var')]==1), ...
    'Missing one of combined_* matrices.');

% Config
TOPK_FOR_PLOTS = 10;   % boxplots & summaries for top-K features by |d|
ALPHA = 0.05;          % significance level

% Convenience: phase matrices and labels
calib.X = [combined_low_calib; combined_high_calib];
calib.y = [zeros(size(combined_low_calib,1),1); ones(size(combined_high_calib,1),1)];
calib.name = 'Calibration';

expd.X = [combined_low_exp; combined_high_exp];
expd.y = [zeros(size(combined_low_exp,1),1); ones(size(combined_high_exp,1),1)];
expd.name = 'Experiment';

% ---------- 0) Helper for Cohen's d (absolute) ----------
cohens_d_abs = @(x1,x2) ...
    (abs(mean(x1,'omitnan') - mean(x2,'omitnan')) ./ ...
    max(eps, sqrt(((numel(x1)-1)*var(x1,0,'omitnan') + (numel(x2)-1)*var(x2,0,'omitnan')) ...
    / max(1, (numel(x1)+numel(x2)-2)))));

% ---------- 1) Phase-level univariate stats + effect sizes ----------
[calib_stats] = phase_univariate(calib.X, calib.y, features, ALPHA);
[exp_stats]   = phase_univariate(expd.X, expd.y, features, ALPHA);

% Save quick tables
writetable(calib_stats, 'extended_calib_univariate.csv');
writetable(exp_stats,   'extended_exp_univariate.csv');

% Identify top features by |d|
[~, idx_d_calib] = sort(calib_stats.Abs_d, 'descend');
[~, idx_d_exp]   = sort(exp_stats.Abs_d,   'descend');

% ---------- 2) Per-subject significance counts & paired across-subject test ----------
have_subject_structs = exist('all_data_calib','var')==1 && exist('all_data_exp','var')==1;
perSubj = struct();

if have_subject_structs
    % Build per-subject LOW/HIGH matrices for both phases (43 features)
    [subjCalibLow, subjCalibHigh, subj_ids_cal] = build_subject_phase(all_data_calib, 'calib');
    [subjExpLow,   subjExpHigh,   subj_ids_exp] = build_subject_phase(all_data_exp,   'exp');

    % Per-subject significance counts (unpaired within subject)
    perSubj.calib_counts = per_subject_significance_counts(subjCalibLow, subjCalibHigh, features, ALPHA, 'Calibration');
    perSubj.exp_counts   = per_subject_significance_counts(subjExpLow,   subjExpHigh,   features, ALPHA, 'Experiment');

    % Save
    writematrix(perSubj.calib_counts, 'per_subject_sig_counts_calib.csv');
    writematrix(perSubj.exp_counts,   'per_subject_sig_counts_exp.csv');

    % Paired across subjects: compare each subject's mean(LOW) vs mean(HIGH)
    paired_calib = paired_across_subjects(subjCalibLow, subjCalibHigh, features, ALPHA);
    paired_exp   = paired_across_subjects(subjExpLow,   subjExpHigh,   features, ALPHA);

    writetable(paired_calib, 'paired_across_subjects_calib.csv');
    writetable(paired_exp,   'paired_across_subjects_exp.csv');
else
    warning('Subject-level structs not found. Skipping per-subject analyses.');
end


% ---------- 3) Correlation heatmaps (Spearman) ----------
plot_corr_heatmaps_native(calib.X, expd.X, features, 'Spearman');  % or 'Pearson'
%

% ---------- 4) For top-K features by |d|, show paired means per subject
topK_calib = idx_d_calib(1:min(TOPK_FOR_PLOTS, numel(idx_d_calib)));
topK_exp   = idx_d_exp(1:min(TOPK_FOR_PLOTS,   numel(idx_d_exp)));

figure(4);
plot_paired_subject_means(subjCalibLow, subjCalibHigh, features, topK_calib, 'Calibration');
figure(5);
plot_paired_subject_means(subjExpLow,   subjExpHigh,   features, topK_exp,   'Experiment');

% ---------- 5) ANOVA F-scores ----------
F_calib = anova_f_scores(calib.X, calib.y);
F_exp   = anova_f_scores(expd.X,   expd.y);
T_F = table((1:numel(features)).', features(:), F_calib(:), F_exp(:), ...
    'VariableNames', {'FeatureID','Feature','Fscore_Calib','Fscore_Exp'});
writetable(T_F, 'anova_fscores_by_phase.csv');

% ---------- 6) Permutation importance (linear SVM, simple split) ----------
perm_calib = permutation_importance_linear_svm(calib.X, calib.y, 10);
perm_exp   = permutation_importance_linear_svm(expd.X,   expd.y,   10);
T_perm = table((1:numel(features)).', features(:), perm_calib(:), perm_exp(:), ...
    'VariableNames', {'FeatureID','Feature','PermImp_Calib','PermImp_Exp'});
writetable(T_perm, 'permutation_importance_by_phase.csv');

% ---------- 7) Effect-size heatmap (Abs d across phases) ----------
% Signed d for both phases
d_calib_signed = signed_cohens_d(calib.X, calib.y);  % [43x1]
d_exp_signed   = signed_cohens_d(expd.X,   expd.y);  % [43x1]
D = [d_calib_signed, d_exp_signed];
pM = [calib_stats.p_value, exp_stats.p_value] < 0.05;   % logical 43x2

% Plot with symmetric limits so colors are comparable
dlim = max(abs(D(:))); if dlim==0, dlim=1; end
figure('Name','Signed Cohen''s d — Calib vs Exp','Color','w');
imagesc(D, [-dlim dlim]); colorbar; axis tight;
set(gca,'XTick',1:2,'XTickLabel',{'Calib','Exp'},'YTick',1:numel(features),'YTickLabel',features);
title('Signed Cohen''s d (HIGH − LOW)');

% Optional: simple red/blue diverging colormap
m = 256; blue=[linspace(0,1,m/2)', linspace(0,1,m/2)', ones(m/2,1)];
red =[ones(m/2,1), linspace(1,0,m/2)', linspace(1,0,m/2)'];
colormap([blue; red]);

% overlay stars
hold on
[rr,cc] = find(pM);
for k=1:numel(rr)
    text(cc(k), rr(k), '*', 'Color','k', 'FontSize',10, 'HorizontalAlignment','center');
end
hold off


%% BLOCK LEVEL ACCURACY PER MODEL

% Initialize
n_blocks = 10;
epochs_per_block = 19;
n_models = 3;


block_accuracies_subjectwise = zeros(n_blocks, n_models, n_subjects);  % [block x model x subjPos]

for subjPos = 1:n_subjects
    subj_num = kept_subjects(subjPos);  % actual ID (e.g., 10)
    sub_exp  = load(sprintf('Subject%d_Results.mat', subj_num));
    exp_log  = sub_exp.experiment_log;

    % Only pre-adaptation epochs
    pred_idxs = isnan([exp_log.adapted_epochs]);
    exp_log   = exp_log(pred_idxs);

    true_label = cell2mat({exp_log.true_label})';

    % Sanity (expects 10 blocks × 19 pre-adapt epochs each)
    assert(length(exp_log) >= n_blocks * epochs_per_block, ...
        'Subject %d does not have enough pre-adapt epochs for %d blocks.', subj_num, n_blocks);

    for b = 1:n_blocks
        idx_range = (1:epochs_per_block) + (b-1)*epochs_per_block;

        stew_preds = cell2mat({exp_log(idx_range).predicted_MWL_stew}');
        heat_preds = cell2mat({exp_log(idx_range).predicted_MWL_heat}');
        matb_preds = cell2mat({exp_log(idx_range).predicted_MWL_matb}');
        gt_labels  = true_label(idx_range);

        block_accuracies_subjectwise(b, 1, subjPos) = mean(stew_preds == gt_labels);
        block_accuracies_subjectwise(b, 2, subjPos) = mean(heat_preds == gt_labels);
        block_accuracies_subjectwise(b, 3, subjPos) = mean(matb_preds == gt_labels);
    end
end

block_accuracies_mean = mean(block_accuracies_subjectwise, 3);

% Ground truth block type
block_labels = [1 0 1 0 1 1 1 0 0 0];

% ----- Plotting -----
figure('Color','w', 'Position',[100 100 900 500]);

% Background MWL blocks
yl = [-10 105];
for b = 1:n_blocks
    if block_labels(b)
        patch([b-0.5 b+0.5 b+0.5 b-0.5], [yl(1) yl(1) yl(2) yl(2)], ...
            [1 0.9 0.9], 'EdgeColor','none');  % High = red tint
    else
        patch([b-0.5 b+0.5 b+0.5 b-0.5], [yl(1) yl(1) yl(2) yl(2)], ...
            [0.9 0.9 1], 'EdgeColor','none');  % Low = blue tint
    end
end
hold on;

% Plot model accuracy curves
x = 1:n_blocks;
y1 = block_accuracies_mean(:,1) * 100;  % STEW
y2 = block_accuracies_mean(:,2) * 100;  % HEAT
y3 = block_accuracies_mean(:,3) * 100;  % MATB

% Define new cool colors
% Slightly brighter Royal Blue
color_STEW = [30, 130, 255] / 255;   % Softer, vivid blue

% Brighter Orange
color_MATB = [255, 160, 40] / 255;   % Warm, more vibrant orange

% Brighter, pink-tinted Purple
color_HEAT = [180, 70, 200] / 255;   % Purple with a touch of magenta

% % Cool and distinct colors
% color_STEW = [110, 40, 140] / 255;   % Darker, deeper violet-purple
% color_MATB   = [30, 130, 255] / 255;   % Clear bright blue
% color_HEAT   = [240, 90, 170] / 255;    % Fresh cyan-turquoise

h1 = plot(x, y1, '-o', 'Color', color_STEW, 'LineWidth', 2, ...
    'MarkerFaceColor', color_STEW);  % STEW
hold on;
h2 = plot(x, y2, '-o', 'Color', color_HEAT, 'LineWidth', 2, ...
    'MarkerFaceColor', color_HEAT);    % HEAT
h3 = plot(x, y3, '-o', 'Color', color_MATB, 'LineWidth', 2, ...
    'MarkerFaceColor', color_MATB); % MATB

yline(50, '--k', 'Chance Level', 'LabelVerticalAlignment','bottom', 'FontSize', 10);

legend([h1 h2 h3], {'STEW','HEAT','MATB'}, 'Location', 'southeastoutside', 'FontSize', 11, 'Box', 'off');

xlabel('Block #', 'FontWeight','bold');
ylabel('Accuracy (%)', 'FontWeight','bold');
title('Per-Block Prediction Accuracy (Mean Across Subjects)', ...
    'FontWeight','bold', 'FontSize', 14);
set(gca, 'FontName','Times New Roman', 'FontSize', 12);
grid on; box off;

% Block annotations
y_bottom = -5;
for b = 1:n_blocks
    txt = ternary(block_labels(b), 'HIGH', 'LOW');
    text(b, y_bottom, txt, 'HorizontalAlignment','center', ...
        'FontSize', 10, 'FontWeight','bold', 'Color',[0.2 0.2 0.2]);
end

xlim([0.5, n_blocks + 0.5]);
xticks(1:n_blocks);
ylim(yl);

% Grid and aesthetics
set(gca, 'YGrid', 'on', 'XGrid', 'off');
box off;



%% Compare Base Model vs. Calib model Accuracy

% Load Base / Pre Calib models
base_stew = load('v1_Base_1000_25wCsp_4sec_proc5_STEW_model.mat');
mdl_workload_STEW = base_stew.mdl;
W_csp_STEW = base_stew.W_csp;                     % Load Training Data Common Spatial Filters

base_heat = load('v1_Base_hyper_1000_25wCsp_4sec_proc5_HEATCHAIR_model.mat');
mdl_workload_HEAT = base_heat.mdl;
W_csp_HEAT = base_heat.W_csp;
best_C_heat = base_heat.best_C;                   % Load Training Data Hyperparameter Best C
best_kernel_heat = base_heat.best_kernel;         % Load Training Data Hyperparameter Best Kernel

base_matb = load('v1_Base_hyper_1000_25wCsp_4sec_proc5_MATB_easy_meddiff_model.mat');
mdl_workload_MATB = base_matb.mdl;
W_csp_MATB = base_matb.W_csp;
best_C_matb = base_matb.best_C;
best_kernel_matb = base_matb.best_kernel;

% Only look at predictions vs true labels before applying "ADAPT"
pred_idxs = isnan([all_data_exp.adapted_epochs]);

% Convert features from cell to matrix
all_epoch_features_STEW = cell2mat({all_data_exp(pred_idxs).STEW_features}');
all_epoch_features_HEAT = cell2mat({all_data_exp(pred_idxs).HEAT_features}');
all_epoch_features_MATB = cell2mat({all_data_exp(pred_idxs).MATB_features}');

true_label = cell2mat({all_data_exp(pred_idxs).true_label});

% Pre Calib
class_labels = {'Low', 'High'};

[pre_acc_stew, class_perf_stew] = eval_mdl_performance(mdl_workload_STEW, all_epoch_features_STEW, true_label, class_labels, 'STEW Model', true);
[pre_acc_heat, class_perf_heat] = eval_mdl_performance(mdl_workload_HEAT, all_epoch_features_HEAT, true_label, class_labels, 'HEAT Model', true);
[pre_acc_matb, class_perf_matb] = eval_mdl_performance(mdl_workload_MATB, all_epoch_features_MATB, true_label, class_labels, 'MATB Model', true);

% If i want to do like this i have to compute the average of all calib models / of their confusion matrices, etc
% and then present one final result
% [pre_acc_stew, class_perf_stew] = eval_mdl_performance(mdl_workload_STEW, all_epoch_features_STEW, true_label, class_labels, 'STEW Model', true);
% [pre_acc_heat, class_perf_heat] = eval_mdl_performance(mdl_workload_HEAT, all_epoch_features_HEAT, true_label, class_labels, 'HEAT Model', true);
% [pre_acc_matb, class_perf_matb] = eval_mdl_performance(mdl_workload_MATB, all_epoch_features_MATB, true_label, class_labels, 'MATB Model', true);

% After Calib Preds
final_preds_stew = [all_data_exp(pred_idxs).predicted_MWL_stew];
final_preds_heat = [all_data_exp(pred_idxs).predicted_MWL_heat];
final_preds_matb = [all_data_exp(pred_idxs).predicted_MWL_matb];

acc_stew = 100 * mean(final_preds_stew == true_label);
acc_heat = 100 * mean(final_preds_heat == true_label);
acc_matb = 100 * mean(final_preds_matb == true_label);

fprintf('\n PRE CALIB ACCURACY:');
fprintf('\n STEW: %.2f%%', pre_acc_stew*100);
fprintf('\n HEAT: %.2f%%', pre_acc_heat*100);
fprintf('\n MATB: %.2f%%', pre_acc_matb*100);

fprintf('\n\n POST CALIB ACCURACY:');
fprintf('\n STEW: %.2f%%', acc_stew);
fprintf('\n HEAT: %.2f%%', acc_heat);
fprintf('\n MATB: %.2f%%', acc_matb);

% Convert to percentages
pre = [pre_acc_stew, pre_acc_heat, pre_acc_matb] * 100;
post = [acc_stew, acc_heat, acc_matb];
delta = post - pre;

model_names = {'STEW', 'HEAT', 'MATB'};

% Overall mean
fprintf('\n\n===== OVERALL MEAN ACCURACY =====\n');
fprintf('PRE  Accuracy : %.2f%%\n', mean(pre));
fprintf('POST Accuracy : %.2f%%\n', mean(post));
fprintf('Delta         : %.2f%%\n', mean(delta));

% Per-Model Accuracy
fprintf('\n===== MODEL-WISE ACCURACY =====\n');
for i = 1:length(model_names)
    fprintf('%s:\n', model_names{i});
    fprintf('  Pre  : %.2f%%\n', pre(i));
    fprintf('  Post : %.2f%%\n', post(i));
    fprintf('  Δ    : %+0.2f%%\n', delta(i));
end

% Sort Models by Post Accuracy (Optional)
[~, sorted_idx] = sort(post, 'descend');
fprintf('\n===== MODEL RANKING (by POST accuracy) =====\n');
for r = 1:length(model_names)
    i = sorted_idx(r);
    fprintf('%d. %s: %.2f%% post accuracy\n', r, model_names{i}, post(i));
end



%% Build pooled confusion matrices from Subject*_Results_Summary.txt

kept_subjects = [1 2 3 4 6 7 8 9 10];
base_dir = 'E:/SchuleJobAusbildung/HTW/MasterThesis/EXPERIMENT/ExperimentData';

models = {'STEW','HEAT','MATB'};
pooled = struct('STEW',zeros(2,2),'HEAT',zeros(2,2),'MATB',zeros(2,2)); % [TP FP; FN TN] with Positive=High

for sid = kept_subjects
    ftxt = fullfile(base_dir, sprintf('Subject_%d',sid), sprintf('Subject%d_Results_Summary.txt',sid));
    if ~exist(ftxt,'file'), warning('Missing %s', ftxt); continue; end
    txt = fileread(ftxt);

    for m = 1:numel(models)
        mdl = models{m};
        blk = extract_block(txt, mdl);

        % Ground-truth & predicted counts by class
        gtL   = str2double(regtok('Low MWL Count\s*:\s*(\d+)\s*\(Ground Truth\)', blk));
        prL   = str2double(regtok('Low MWL Count\s*:\s*\d+\s*\(Ground Truth\),\s*(\d+)\s*\(Predicted\)', blk));
        gtH   = str2double(regtok('High MWL Count\s*:\s*(\d+)\s*\(Ground Truth\)', blk));
        prH   = str2double(regtok('High MWL Count\s*:\s*\d+\s*\(Ground Truth\),\s*(\d+)\s*\(Predicted\)', blk));

        precH = str2double(regtok('Precision \(High\)\s*:\s*([0-9.]+)\s*%', blk));  % %
        recH  = str2double(regtok('Recall \(High\)\s*:\s*([0-9.]+)\s*%', blk));     % %
        acc   = str2double(regtok([mdl '\s+Accuracy\s*:\s*([0-9.]+)\s*%'], blk));   % %

        % Reconstruct per-subject confusion matrix (High=positive)
        [TP,FP,FN,TN] = cm_from_counts(gtH,gtL,prH,prL,precH,recH,acc);

        pooled.(mdl) = pooled.(mdl) + [TP FP; FN TN];
    end
end

% ---- Print pooled matrices + metrics
print_cm('STEW', pooled.STEW);
print_cm('HEAT', pooled.HEAT);
print_cm('MATB', pooled.MATB);

% Per-model matrices you posted (High=positive):
C_stew_pf = [369 327; 486 528];   % [TP FP; FN TN]
C_heat_pf = [455 329; 400 526];
C_matb_pf = [494 324; 361 531];


%% === Individual model confusion matrices (with requested settings) ===
% Use your existing pooled matrices [TP FP; FN TN] (High = positive)
if exist('pooled','var') && isfield(pooled,'STEW')
    C_STEW_pf = pooled.STEW;
    C_HEAT_pf = pooled.HEAT;
    C_MATB_pf = pooled.MATB;
else
    % or your previously saved variables:
    C_STEW_pf = C_stew_pf;
    C_HEAT_pf = C_heat_pf;
    C_MATB_pf = C_matb_pf;
end

class_labels = {'High','Low'};  % rows = true, cols = predicted

plot_conf_chart(C_STEW_pf, class_labels, 'STEW (pooled)');
plot_conf_chart(C_HEAT_pf, class_labels, 'HEAT (pooled)');
plot_conf_chart(C_MATB_pf, class_labels, 'MATB (pooled)');

% Sum (micro average)
C_sum_pf  = C_stew_pf + C_heat_pf + C_matb_pf;      % [TP FP; FN TN]

% Reorder to standard confusionchart layout: [ [TP FN]; [FP TN] ]
C_plot = [ C_sum_pf(1,1)  C_sum_pf(2,1) ;   % [TP  FN]
           C_sum_pf(1,2)  C_sum_pf(2,2) ];  % [FP  TN]

% Metrics
TP = C_sum_pf(1,1); FP = C_sum_pf(1,2); FN = C_sum_pf(2,1); TN = C_sum_pf(2,2);
prec = TP/(TP+FP); rec = TP/(TP+FN);
f1   = 2*prec*rec/(prec+rec); acc = (TP+TN)/(TP+FP+FN+TN);

% Plot
figure('Color','w');
cc = confusionchart(C_plot, {'High','Low'}, ...
    'RowSummary','row-normalized','ColumnSummary','column-normalized');
title(sprintf('Combined Models (micro) — Acc %.2f%%  Prec %.2f%%  Rec %.2f%%  F1 %.2f%%', ...
      acc*100, prec*100, rec*100, f1*100));



%% Plot or calculate the Majority Vote Buffer Accuracy

% Compute Total Valid Majority Vote Buffer Accuracy (valid + the moment the
% adapt command is sent
% Where mwl is not NaN in .correct:
majority_correct = ~isnan([all_data_exp.correct]);
valid_majority_correct = find(majority_correct);        % Get the positions/ epoch idxs

% including the one where the adaptation is sent
all_corrections = cellfun(@(x) x, {all_data_exp.adapt_command});
adapt_idxs = find(strcmp(all_corrections, "HIGH") | strcmp(all_corrections, "LOW"));

% Get all correct epoch idxs or the adapt_idxs
combined_idxs = sort([valid_majority_correct, adapt_idxs]);

% Get majority vote buffer predictions
majority_preds = [all_data_exp(combined_idxs).majority_MWL];

% Get corresponding ground truth labels
true_labels = [all_data_exp(combined_idxs).true_label];

% Compute Accuracy
total_majority_buffer_accuracy = 100 * mean(majority_preds == true_labels);
fprintf('\n\nMajority MWL Buffer Accuracy : %.2f %%\n', total_majority_buffer_accuracy);


%% Plot or calculate the Total Block level accuracy (x/10 blocks per Experiment)

base_sequence = [1,0,1,0,1,1,1,0,0,0];

% Concatenate the base sequence for each subject
sequence = repmat(base_sequence, 1, n_subjects);

% Compute BLOCK LEVEL accuracy x / numBlocks
% Extract all non-empty correction commands
all_corrections = {all_data_exp.adapt_command};
adapt_idxs = find(~cellfun(@isempty, all_corrections));   % Indices where ADAPT command was sent
correction_cmds = all_corrections(adapt_idxs);            % Extract the commands in order

% Convert commands to binary values
predicted_block_labels = zeros(1, length(correction_cmds));
for i = 1:length(correction_cmds)
    if strcmpi(correction_cmds{i}, 'HIGH')
        predicted_block_labels(i) = 1;
    elseif strcmpi(correction_cmds{i}, 'LOW')
        predicted_block_labels(i) = 0;
    else
        predicted_block_labels(i) = NaN;
    end
end

% Remove NaNs from predicted_block_labels and match indices in sequence
valid_idx = ~isnan(predicted_block_labels);
predicted_block_labels = predicted_block_labels(valid_idx);

% Ground truth block labels from sequence
true_block_labels = sequence(1:length(predicted_block_labels));  % make sure same length

% Compute block-level accuracy
block_level_accuracy = mean(predicted_block_labels == true_block_labels) * 100;
fprintf('\n\nADAPT Command Block-Level Accuracy : %.2f %%\n', block_level_accuracy);


%% Subject Summary Sheet Evaluations

% Aggregate metrics from Subject*_Results_Summary.txt
% Configure which subjects to include
kept_subjects = [1 2 3 4 6 7 8 9 10];   % removed: 5 and 11


% Preallocate storage
nS = numel(kept_subjects);
SIDs = kept_subjects(:);

% Per-subject metrics (all stored as numeric percentages, except time in sec)
T = table('Size',[nS 1], 'VariableTypes',"double", 'VariableNames',"SID");
T.SID = SIDs;

vars = { ...
    'AvgProcTime_s', ...
    'STEW_Prec','STEW_Recall','STEW_F1','STEW_Acc', ...
    'HEAT_Prec','HEAT_Recall','HEAT_F1','HEAT_Acc', ...
    'MATB_Prec','MATB_Recall','MATB_F1','MATB_Acc', ...
    'Global_Prec','Global_Recall','Global_F1','Total_Acc', ...
    'Majority_Buffer_Acc','ADAPT_Block_Acc'};

for v = vars, T.(v{1}) = nan(nS,1); end

% Helpers
extract_num = @(pat,txt) local_extract_num(pat, txt);  % returns double or NaN

for k = 1:nS

    sid = kept_subjects(k);
    results_dir = sprintf('E:/SchuleJobAusbildung/HTW/MasterThesis/EXPERIMENT/ExperimentData/Subject_%d', sid);
    ftxt = fullfile(results_dir, sprintf('Subject%d_Results_Summary.txt', sid));
    if ~exist(ftxt,'file')
        warning('Missing file: %s', ftxt);
        continue;
    end
    txt = fileread(ftxt);

    % -------- General --------
    T.AvgProcTime_s(k)      = extract_num('Avg Process Time\s*:\s*([0-9.]+)\s*sec', txt);

    % -------- carve model sections once --------
    stewBlk = get_section(txt, 'STEW', 'HEAT');
    heatBlk = get_section(txt, 'HEAT', 'MATB');
    matbBlk = get_section(txt, 'MATB', 'GLOBAL PERFORMANCE');

    % -------- STEW (from its own block) --------
    T.STEW_Prec(k)   = extract_num('Precision \(High\)\s*:\s*([0-9.]+)\s*%', stewBlk);
    T.STEW_Recall(k) = extract_num('Recall \(High\)\s*:\s*([0-9.]+)\s*%',    stewBlk);
    T.STEW_F1(k)     = extract_num('F1-Score \(High\)\s*:\s*([0-9.]+)\s*%',  stewBlk);
    T.STEW_Acc(k)    = extract_num('STEW Accuracy\s*:\s*([0-9.]+)\s*%',      stewBlk);

    % -------- HEAT (from its own block) --------
    T.HEAT_Prec(k)   = extract_num('Precision \(High\)\s*:\s*([0-9.]+)\s*%', heatBlk);
    T.HEAT_Recall(k) = extract_num('Recall \(High\)\s*:\s*([0-9.]+)\s*%',    heatBlk);
    T.HEAT_F1(k)     = extract_num('F1-Score \(High\)\s*:\s*([0-9.]+)\s*%',  heatBlk);
    T.HEAT_Acc(k)    = extract_num('HEAT Accuracy\s*:\s*([0-9.]+)\s*%',      heatBlk);

    % -------- MATB (from its own block) --------
    T.MATB_Prec(k)   = extract_num('Precision \(High\)\s*:\s*([0-9.]+)\s*%', matbBlk);
    T.MATB_Recall(k) = extract_num('Recall \(High\)\s*:\s*([0-9.]+)\s*%',    matbBlk);
    T.MATB_F1(k)     = extract_num('F1-Score \(High\)\s*:\s*([0-9.]+)\s*%',  matbBlk);
    T.MATB_Acc(k)    = extract_num('MATB Accuracy\s*:\s*([0-9.]+)\s*%',      matbBlk);

    % -------- Global --------
    T.Global_Prec(k)        = extract_num('Global Precision \(High\)\s*:\s*([0-9.]+)\s*%', txt);
    T.Global_Recall(k)      = extract_num('Global Recall \(High\)\s*:\s*([0-9.]+)\s*%', txt);
    T.Global_F1(k)          = extract_num('Global F1-Score \(High\)\s*:\s*([0-9.]+)\s*%', txt);
    T.Total_Acc(k)          = extract_num('Total Combined Accuracy\s*:\s*([0-9.]+)\s*%', txt);

    % -------- Majority vote / Block accuracy --------
    T.Majority_Buffer_Acc(k)= extract_num('Majority MWL Buffer Accuracy\s*:\s*([0-9.]+)\s*%', txt);
    T.ADAPT_Block_Acc(k)    = extract_num('ADAPT Command Block-Level Accuracy\s*:\s*([0-9.]+)\s*%', txt);
end

disp('Per-subject parsed metrics:');
disp(T);

% Macro-averages (simple mean across subjects, ignoring NaNs)
M = varfun(@(x) mean(x,'omitnan'), T, 'InputVariables', T.Properties.VariableNames(2:end));
M.Properties.RowNames = {'MeanAcrossSubjects'};
disp('Macro-averages (mean across included subjects):');
disp(M);

% Pretty print a small summary
fprintf('\n==== Macro-Averages Across Subjects ====\n');
fprintf('Avg Process Time (sec)          : %.3f\n', M.Fun_AvgProcTime_s);
avgProcTime_std = std(T.AvgProcTime_s, 'omitnan');
fprintf('Avg Process Time (sec)          : %.3f ± %.3f\n', ...
        M.Fun_AvgProcTime_s, avgProcTime_std);
fprintf('\nSTEW   — Prec/Recall/F1/Acc (%%) : %.2f / %.2f / %.2f / %.2f\n', ...
    M.Fun_STEW_Prec, M.Fun_STEW_Recall, M.Fun_STEW_F1, M.Fun_STEW_Acc);
fprintf('HEAT   — Prec/Recall/F1/Acc (%%) : %.2f / %.2f / %.2f / %.2f\n', ...
    M.Fun_HEAT_Prec, M.Fun_HEAT_Recall, M.Fun_HEAT_F1, M.Fun_HEAT_Acc);
fprintf('MATB   — Prec/Recall/F1/Acc (%%) : %.2f / %.2f / %.2f / %.2f\n', ...
    M.Fun_MATB_Prec, M.Fun_MATB_Recall, M.Fun_MATB_F1, M.Fun_MATB_Acc);
fprintf('\nGLOBAL — Prec/Recall/F1/Acc (%%) : %.2f / %.2f / %.2f / %.2f\n', ...
    M.Fun_Global_Prec, M.Fun_Global_Recall, M.Fun_Global_F1, M.Fun_Total_Acc);
fprintf('Majority Buffer Acc (%%)         : %.2f\n', M.Fun_Majority_Buffer_Acc);
fprintf('ADAPT Block-Level Acc (%%)       : %.2f\n', M.Fun_ADAPT_Block_Acc);

% Optional: save per-subject table and means
writetable(T, 'per_subject_summary_metrics.csv');
writetable(stack(M, M.Properties.VariableNames, 'NewDataVariableName','Mean', 'IndexVariableName','Metric'), ...
    'macro_averages_summary_metrics.csv');





%% ORDER EFFECT: 1-way ANOVA - not averaged across subjects -> order_pairs x subject_number amount of ANOVA inputs

% -----------------
% For Calibration TOTAL
n_blocks = 20;
epochs_per_block = 30;
n_features = 43;

order_labels_all_calib = {};
subject_labels_all_calib = {};
feature_values_all_calib = cell(n_features, 1);
for f = 1:n_features, feature_values_all_calib{f} = []; end

for s = 1:n_subjects
    % Sequentially take this subject’s calibration epochs
    start_idx = (s-1) * n_blocks * epochs_per_block + 1;
    end_idx   = s * n_blocks * epochs_per_block;
    subj_data = all_data_calib(start_idx:end_idx);

    % Compute block means for 1..20
    block_means = zeros(n_blocks, n_features);
    for b = 1:n_blocks
        block_start = (b-1)*epochs_per_block + 1;
        block_end   = b*epochs_per_block;
        block_data  = subj_data(block_start:block_end);

        block_feats = zeros(epochs_per_block, n_features);
        for e = 1:epochs_per_block
            base     = block_data(e).features_STEW(1:25);
            stew_csp = block_data(e).features_STEW(26:31);
            heat_csp = block_data(e).features_HEAT(26:31);
            matb_csp = block_data(e).features_MATB(26:31);
            block_feats(e, :) = [base, stew_csp, heat_csp, matb_csp];
        end
        block_means(b, :) = mean(block_feats, 1);
    end

    % Sequence for calibration total
    seq_calib_total = [1 1 0 0 0 1 1 0 1 0 1 0 0 1 1 1 1 0 0 0];

    % Build order pairs
    for p = 1:(n_blocks-1)
        order_str = sprintf('%d%d', seq_calib_total(p), seq_calib_total(p+1));
        pair_mean = mean([block_means(p,:); block_means(p+1,:)], 1);
        for f = 1:n_features
            feature_values_all_calib{f}(end+1, 1) = pair_mean(f);
        end
        order_labels_all_calib{end+1, 1}  = order_str;
        subject_labels_all_calib{end+1, 1} = sprintf('S%d', kept_subjects(s));
    end
end

pvals_oneway_calib = zeros(n_features, 1);
for f = 1:n_features
    feature_vals_calib = feature_values_all_calib{f};
    pvals_oneway_calib(f) = anova1(feature_vals_calib, order_labels_all_calib, 'off');
end


% -----------------
% For Calibration USED (blocks 13..20 only)
n_blocks = 8;  % analyzing 8 blocks
epochs_per_block = 30;
n_features = 43;

order_labels_all_calib_used = {};
subject_labels_all_calib_used = {};
feature_values_all_calib_used = cell(n_features, 1);
for f = 1:n_features, feature_values_all_calib_used{f} = []; end

for s = 1:n_subjects
    % Sequentially take this subject’s calibration epochs
    start_idx = (s-1) * 20 * epochs_per_block + 1;  % full 20 blocks per subject
    end_idx   = s * 20 * epochs_per_block;
    subj_data = all_data_calib(start_idx:end_idx);

    % Only keep blocks 13..20
    used_blocks = 13:20;
    block_means = zeros(numel(used_blocks), n_features);
    for bi = 1:numel(used_blocks)
        b = used_blocks(bi);
        block_start = (b-1)*epochs_per_block + 1;
        block_end   = b*epochs_per_block;
        block_data  = subj_data(block_start:block_end);

        block_feats = zeros(epochs_per_block, n_features);
        for e = 1:epochs_per_block
            base     = block_data(e).features_STEW(1:25);
            stew_csp = block_data(e).features_STEW(26:31);
            heat_csp = block_data(e).features_HEAT(26:31);
            matb_csp = block_data(e).features_MATB(26:31);
            block_feats(e, :) = [base, stew_csp, heat_csp, matb_csp];
        end
        block_means(bi, :) = mean(block_feats, 1);
    end

    % Sequence for calibration USED (last 8 blocks)
    seq_calib_total = [1 1 0 0 0 1 1 0 1 0 1 0 0 1 1 1 1 0 0 0];
    seq_calib_used  = seq_calib_total(end-7:end);

    % Build order pairs for the 8 used blocks
    for p = 1:(n_blocks-1)
        order_str = sprintf('%d%d', seq_calib_used(p), seq_calib_used(p+1));
        pair_mean = mean([block_means(p,:); block_means(p+1,:)], 1);
        for f = 1:n_features
            feature_values_all_calib_used{f}(end+1, 1) = pair_mean(f);
        end
        order_labels_all_calib_used{end+1, 1}  = order_str;
        subject_labels_all_calib_used{end+1, 1} = sprintf('S%d', kept_subjects(s));
    end
end

pvals_oneway_calib_used = zeros(n_features, 1);
for f = 1:n_features
    feature_vals_calib_used = feature_values_all_calib_used{f};
    pvals_oneway_calib_used(f) = anova1(feature_vals_calib_used, order_labels_all_calib_used, 'off');
end

% -----------------
% EXPERIMENT (pre-ADAPT only)
n_features = 43;
n_blocks = 10;
epochs_per_block = 19;

order_labels_all_exp = {};
subject_labels_all_exp = {};
feature_values_all_exp = cell(n_features, 1);

% Pre-ADAPT epochs only
pred_idxs     = isnan([all_data_exp.adapted_epochs]);
pre_adapt_exp = all_data_exp(pred_idxs);

for f = 1:n_features, feature_values_all_exp{f} = []; end

for s = 1:n_subjects
    subj_data = pre_adapt_exp((s-1)*n_blocks*epochs_per_block + 1 : s*n_blocks*epochs_per_block);

    % Precompute block means for all features
    block_means = zeros(n_blocks, n_features);
    for b = 1:n_blocks
        block_start = (b-1)*epochs_per_block + 1;
        block_end   = b*epochs_per_block;

        block_feats = zeros(epochs_per_block, n_features);
        for e = 1:epochs_per_block
            epoch = subj_data(block_start + e - 1);

            base     = epoch.STEW_features(1:25);
            stew_csp = epoch.STEW_features(26:31);
            heat_csp = epoch.HEAT_features(26:31);
            matb_csp = epoch.MATB_features(26:31);

            block_feats(e, :) = [base, stew_csp, heat_csp, matb_csp];
        end
        block_means(b, :) = mean(block_feats, 1);
    end

    % Fixed experiment sequence (10 blocks -> 9 pairs)
    seq_exp = [1 0 1 0 1 1 1 0 0 0];

    % Build order pairs
    for p = 1:(n_blocks-1)
        order_str = sprintf('%d%d', seq_exp(p), seq_exp(p+1));
        pair_mean = mean([block_means(p,:); block_means(p+1,:)], 1);

        for f = 1:n_features
            feature_values_all_exp{f}(end+1, 1) = pair_mean(f);
        end
        order_labels_all_exp{end+1, 1}  = order_str;
        subject_labels_all_exp{end+1, 1} = sprintf('S%d', kept_subjects(s));
    end
end

pvals_oneway_exp = zeros(n_features, 1);
for f = 1:n_features
    feature_vals_exp = feature_values_all_exp{f};
    pvals_oneway_exp(f) = anova1(feature_vals_exp, order_labels_all_exp, 'off');
end


% -----------------
% Significance (raw, Bonferroni, Holm)
alpha = 0.05;
n_features = 43;
alpha_bonf = alpha / n_features;

sig_feats_oneway_raw_calib = find(pvals_oneway_calib < alpha);
sig_feats_oneway_raw_used  = find(pvals_oneway_calib_used < alpha);
sig_feats_oneway_raw_exp   = find(pvals_oneway_exp < alpha);

sig_feats_oneway_bonf_calib = find(pvals_oneway_calib < alpha_bonf);
sig_feats_oneway_bonf_used  = find(pvals_oneway_calib_used < alpha_bonf);
sig_feats_oneway_bonf_exp   = find(pvals_oneway_exp < alpha_bonf);

sig_feats_oneway_holm_calib = find(holm_bonferroni(pvals_oneway_calib, alpha));
sig_feats_oneway_holm_used  = find(holm_bonferroni(pvals_oneway_calib_used, alpha));
sig_feats_oneway_holm_exp   = find(holm_bonferroni(pvals_oneway_exp, alpha));

fprintf('\n--- One-way ANOVA (non-corrected) ---\n');
print_features('Calib Total', sig_feats_oneway_raw_calib);
print_features('Calib Used',  sig_feats_oneway_raw_used);
print_features('Experiment',  sig_feats_oneway_raw_exp);

fprintf('\n--- One-way ANOVA (Bonferroni-corrected) ---\n');
print_features('Calib Total', sig_feats_oneway_bonf_calib);
print_features('Calib Used',  sig_feats_oneway_bonf_used);
print_features('Experiment',  sig_feats_oneway_bonf_exp);

fprintf('\n--- One-way ANOVA (Holm-Bonferroni-corrected) ---\n');
print_features('Calib Total', sig_feats_oneway_holm_calib);
print_features('Calib Used',  sig_feats_oneway_holm_used);
print_features('Experiment',  sig_feats_oneway_holm_exp);


%%
% Helper inline ternary function
function out = ternary(cond, val_true, val_false)
if cond
    out = val_true;
else
    out = val_false;
end
end


function x = exp_combine43(epoch)
base     = epoch.STEW_features(1:25);
stew_csp = epoch.STEW_features(26:31);
heat_csp = epoch.HEAT_features(26:31);
matb_csp = epoch.MATB_features(26:31);
x = [base, stew_csp, heat_csp, matb_csp];  % 1×43
end

function x = calib_combine43(epoch)
base     = epoch.features_STEW(1:25);
stew_csp = epoch.features_STEW(26:31);
heat_csp = epoch.features_HEAT(26:31);
matb_csp = epoch.features_MATB(26:31);
x = [base, stew_csp, heat_csp, matb_csp];  % 1×43
end


% Holm Bonferroni Correction Helper
function sig_idx = holm_bonferroni(p_vals, alpha)
if nargin<2, alpha=0.05; end
[sp, ord] = sort(p_vals(:));
m = numel(sp);
keep = false(m,1);
for i=1:m
    if sp(i) <= alpha/(m-i+1)
        keep(i)=true;
    else
        break
    end
end
sig_idx = false(size(p_vals));
sig_idx(ord(keep)) = true;
end


% Build long tables (Subject, OrderType, FeatureValue) and run mixed models
function p_vals = rm_anova_order(pairs_cell, order_types_cell, n_subjects, n_features)
% pairs_cell: cell{s} = [P_s × 43] pair means for subject s (P_s may vary)
% order_types_cell: 1×P cellstr with order type per pair index, same for all subjects in your design
p_vals = nan(n_features,1);
% Build long table across subjects
for f = 1:n_features
    subj = []; ord  = {}; y = [];
    for s = 1:n_subjects
        M = pairs_cell{s};               % [P × 43]
        P = size(M,1);
        subj = [subj; repmat(s,P,1)];
        ord  = [ord;  order_types_cell(1:P)'];  % ensure column cellstr
        y    = [y;    M(:,f)];
    end
    T = table(subj, categorical(ord), y, 'VariableNames', ...
        {'Subject','OrderType','FeatureValue'});
    % Mixed model: random intercept per subject, fixed OrderType
    lme = fitlme(T, 'FeatureValue ~ OrderType + (1|Subject)');
    A   = anova(lme, 'DFMethod','Satterthwaite');   % robust df
    % extract p for OrderType (row where Term == 'OrderType')
    row = find(strcmp(A.Term,'OrderType'), 1);
    p_vals(f) = A.pValue(row);
end
end


function print_features(label, feats)
if isempty(feats)
    fprintf('%-15s Significant features: []\n', label);
else
    fprintf('%-15s Significant features: %s\n', label, mat2str(feats(:)'));
end
end


function pvals = run_rm_for_phase(feature_values_all, order_labels_all, subject_labels_all, n_features)
% Run mixed model FeatureValue ~ OrderType + (1|Subject)
pvals = nan(n_features,1);
% ensure cell columns
if isrow(order_labels_all),  order_labels_all  = order_labels_all'; end
if isrow(subject_labels_all),subject_labels_all= subject_labels_all'; end

for f = 1:n_features
    y = feature_values_all{f};
    T = table(y, categorical(order_labels_all), categorical(subject_labels_all), ...
        'VariableNames', {'FeatureValue','OrderType','Subject'});
    % Fit mixed model
    try
        lme = fitlme(T, 'FeatureValue ~ OrderType + (1|Subject)');
        A   = anova(lme, 'DFMethod','Satterthwaite'); % more accurate dfs
    catch
        % Fallback if DFMethod not available
        lme = fitlme(T, 'FeatureValue ~ OrderType + (1|Subject)');
        A   = anova(lme);
    end
    row = find(strcmp(A.Term,'OrderType'), 1);
    pvals(f) = A.pValue(row);
end
end

function d = cohen_d(x1, x2)
n1 = numel(x1); n2 = numel(x2);
% sample variances (n-1) so pooled SD matches the classic Cohen's d formula
s1 = var(x1, 0);
s2 = var(x2, 0);
sp = sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1 + n2 - 2));
d  = (mean(x1) - mean(x2)) / sp;
end



% Function to run analysis for a phase
function resultsTable = analyze_phase(low_data, high_data, features)
nFeat = size(low_data, 2);
results = cell(nFeat, 11); % columns: Feature, meanL, stdL, meanH, stdH, normL, normH, Test, p, d_signed, StatEffect

for f = 1:nFeat
    low_vals  = low_data(:, f);
    high_vals = high_data(:, f);

    % Means and stds
    mL = mean(low_vals);
    sL = std(low_vals);
    mH = mean(high_vals);
    sH = std(high_vals);

    % Normality checks (accept if at least one passes)
    normL_ad  = adtest(low_vals) == 0;
    normL_lil = lillietest(low_vals) == 0;
    normH_ad  = adtest(high_vals) == 0;
    normH_lil = lillietest(high_vals) == 0;
    normalL = normL_ad || normL_lil;
    normalH = normH_ad || normH_lil;

    % Decide test
    if normalL && normalH
        [~, p] = ttest2(low_vals, high_vals);
        testType = 't-test';
    else
        p = ranksum(low_vals, high_vals);
        testType = 'Mann-Whitney';
    end

    % Cohen's d (absolute)
    n1 = numel(low_vals);
    n2 = numel(high_vals);
    sp = sqrt(((n1-1)*var(low_vals,0) + (n2-1)*var(high_vals,0)) / (n1+n2-2));
    if sp == 0 || ~isfinite(sp)
        d_abs = NaN;
    else
        d_abs = abs((mean(low_vals) - mean(high_vals)) / sp);
    end

    % StatEffect category
    if p >= 0.05
        statEffect = "not-sig";
    elseif d_abs >= 0.8
        statEffect = "sig-large";
    elseif d_abs >= 0.5
        statEffect = "sig-medium";
    elseif d_abs >= 0.2
        statEffect = "sig-small";
    elseif d_abs > 0
        statEffect = "sig-tiny";
    else
        statEffect = "not-sig";
    end

    % Store results
    results(f,:) = {features{f}, mL, sL, mH, sH, normalL, normalH, testType, p, d_abs, statEffect};
end

% Convert to table
resultsTable = cell2table(results, ...
    'VariableNames', {'Feature','Mean_Low','Std_Low','Mean_High','Std_High', ...
    'Normal_Low','Normal_High','Test_Type','p_value','Cohens_d','StatEffect'});
end


%% -------- Global descriptive features analysis (calib & exp) --------
function T = compute_phase_descriptives(X, feature_names, outfile, phase_label)
% X: [nSamples x nFeatures] for a phase (LOW & HIGH concatenated)
% feature_names: 1xN cellstr
% outfile: CSV filename to save
% phase_label: string for printing

% Guard against empty or mismatched inputs
if isempty(X)
    error('Input data for %s is empty.', phase_label);
end
if size(X,2) ~= numel(feature_names)
    warning('Number of columns in X (%d) != number of feature names (%d).', size(X,2), numel(feature_names));
    % Create fallback names if needed
    feature_names = arrayfun(@(k) sprintf('Feature%d', k), 1:size(X,2), 'UniformOutput', false);
end

% Per-feature stats (column-wise), NaN-safe
mu  = mean(X, 'omitnan');              % 1 x nFeatures
sd  = std(X, 0, 'omitnan');            % sample std
mn  = min(X, [], 1, 'omitnan');
mx  = max(X, [], 1, 'omitnan');
N   = sum(~isnan(X), 1);               % count of non-NaN per feature

% Build table
T = table( (1:size(X,2)).', feature_names(:), mu(:), sd(:), mn(:), mx(:), N(:), ...
    'VariableNames', {'FeatureID','Feature','Mean','Std','Min','Max','N'});

% Save
writetable(T, outfile);

% Global (phase-level) summary across all values in X
allvals = X(:);
gMean = mean(allvals, 'omitnan');
gStd  = std(allvals, 0, 'omitnan');
gMin  = min(allvals, [], 'omitnan');
gMax  = max(allvals, [], 'omitnan');

fprintf('\n[%s] Global descriptive summary across ALL features & samples:\n', phase_label);
fprintf('  Global mean = %.3f\n', gMean);
fprintf('  Global std  = %.3f\n', gStd);
fprintf('  Global min  = %.3f\n', gMin);
fprintf('  Global max  = %.3f\n', gMax);
fprintf('  Saved per-feature stats to: %s\n\n', outfile);
end


function counts = per_subject_significance_counts(subjLow, subjHigh, features, alpha, phase_name)
nS = numel(subjLow); nF = size(subjLow{1},2);
counts = zeros(nF,1);
for f = 1:nF
    sig_c = 0;
    for s = 1:nS
        x0 = subjLow{s}(:,f); x1 = subjHigh{s}(:,f);
        if numel(x0)<3 || numel(x1)<3, continue; end
        norm0 = (adtest(x0)==0) || (lillietest(x0)==0);
        norm1 = (adtest(x1)==0) || (lillietest(x1)==0);
        if norm0 && norm1, [~,p]=ttest2(x0,x1); else, p=ranksum(x0,x1); end
        if p<alpha, sig_c=sig_c+1; end
    end
    counts(f)=sig_c;
end
figure('Name',['Per-subject Significance Counts — ',phase_name],'Color','w');
bar(counts);
set(gca,'XTick',1:nF,'XTickLabel',features,'XTickLabelRotation',90);
ylabel('#Subjects with p<0.05'); title(['Per-subject significance counts — ', phase_name]);
grid on; box on;
end

function T = paired_across_subjects(subjLow, subjHigh, features, alpha)
% For each feature: within each subject, mean(LOW) and mean(HIGH),
% then paired test across subjects on the difference vector.
nS = numel(subjLow);
nF = size(subjLow{1},2);
out = cell(nF,5);
for f = 1:nF
    m0 = nan(nS,1); m1 = nan(nS,1);
    for s = 1:nS
        m0(s) = mean(subjLow{s}(:,f),'omitnan');
        m1(s) = mean(subjHigh{s}(:,f),'omitnan');
    end
    diffv = m1 - m0; % HIGH - LOW per subject

    % Normality on differences
    normDiff = (adtest(diffv)==0) || (lillietest(diffv)==0);
    if normDiff
        [~, p, ~, stats] = ttest(m1, m0); % paired t-test
        testName = 'paired t';
        effect   = mean(diffv,'omitnan') / std(diffv,0,'omitnan'); % standardized mean diff (paired d)
    else
        p = signrank(m1, m0);
        testName = 'Wilcoxon signed-rank';
        effect   = median(diffv,'omitnan'); % report median diff as effect-like summary
    end
    out(f,:) = {features{f}, testName, p, mean(diffv,'omitnan'), effect};
end
T = cell2table(out, 'VariableNames', {'Feature','Test','p_value','MeanDiff_HIGH_minus_LOW','EffectSummary'});
end

function plot_corr_heatmaps_native(Xcal, Xexp, featnames, corrType)
if nargin<4, corrType = 'Spearman'; end
assert(size(Xcal,2)==numel(featnames) && size(Xexp,2)==numel(featnames), ...
    'Feature count mismatch vs featnames.');

Rcal = corr(Xcal, 'Type',corrType, 'Rows','pairwise');
Rexp = corr(Xexp, 'Type',corrType, 'Rows','pairwise');
Rdel = Rexp - Rcal;

figure('Name',['Correlation Heatmaps (',corrType,') — native order'],'Color','w');
t = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

nexttile; imagesc(Rcal, [-1 1]); axis image; title('Calibration');
set(gca,'XTick',1:numel(featnames),'XTickLabel',featnames,'XTickLabelRotation',90, ...
    'YTick',1:numel(featnames),'YTickLabel',featnames); colorbar;

nexttile; imagesc(Rexp, [-1 1]); axis image; title('Correlation (Spearman) - Experiment Features');
set(gca,'XTick',1:numel(featnames),'XTickLabel',featnames,'XTickLabelRotation',90, ...
    'YTick',1:numel(featnames),'YTickLabel',featnames); colorbar;

nexttile;
clim = max(abs(Rdel(:))); if clim==0, clim=1; end
imagesc(Rdel, [-clim clim]); axis image; title('\Delta corr (Exp − Calib)');
set(gca,'XTick',1:numel(featnames),'XTickLabel',featnames,'XTickLabelRotation',90, ...
    'YTick',1:numel(featnames),'YTickLabel',featnames); colorbar;
%title(t, ['Correlation (',corrType,') — Experiment']);
end

function plot_condition_boxplots(X, y, features, feat_idx, phase_name)
% Build grouped arrays and boxplot per feature
nK = numel(feat_idx);
for k = 1:nK
    f = feat_idx(k);
    x0 = X(y==0, f);
    x1 = X(y==1, f);
    subplot(ceil(nK/2), 2, k);
    boxplot([x0; x1], [zeros(size(x0)); ones(size(x1))], 'Labels',{'LOW','HIGH'});
    title(sprintf('%s — %s', phase_name, features{f}));
    ylabel('Value'); grid on; box on;
end
end

function F = anova_f_scores(X, y)
% One-vs-one: between/within variance ratio for 2-class case
nF = size(X,2);
F = zeros(nF,1);
x0 = X(y==0,:); x1 = X(y==1,:);
m0 = mean(x0,1,'omitnan'); m1 = mean(x1,1,'omitnan');
mG = mean(X,1,'omitnan');
n0 = sum(y==0); n1 = sum(y==1);

SSB = n0*(m0 - mG).^2 + n1*(m1 - mG).^2;                  % between
SSW = sum((x0 - m0).^2,'omitnan') + sum((x1 - m1).^2,'omitnan'); % within
F = (SSB ./ max(eps, SSW)).';
end

function perm = permutation_importance_linear_svm(X, y, n_repeats)
if nargin<3, n_repeats=10; end
% Simple holdout to keep things quick
cv = cvpartition(y,'Holdout',0.3);
Xtr = X(training(cv),:); ytr = y(training(cv));
Xte = X(test(cv),:);     yte = y(test(cv));

mdl = fitcsvm(Xtr,ytr,'KernelFunction','linear','Standardize',false);
yhat = predict(mdl, Xte);
acc0 = mean(yhat==yte);

nF = size(X,2);
perm = zeros(nF,1);
for f = 1:nF
    drops = zeros(n_repeats,1);
    for r = 1:n_repeats
        Xperm = Xte;
        Xperm(:,f) = Xperm(randperm(size(Xperm,1)), f);
        yhatp = predict(mdl, Xperm);
        accp  = mean(yhatp==yte);
        drops(r) = acc0 - accp;
    end
    perm(f) = mean(drops);
end
end

function T = phase_univariate(X, y, features, alpha)
% Univariate: means/stds, normality (AD or Lillie pass), t-test vs ranksum, |d|
nF = size(X,2);
out = cell(nF, 12);
for f = 1:nF
    xf = X(:,f);
    x0 = xf(y==0); x1 = xf(y==1);

    m0 = mean(x0,'omitnan'); s0 = std(x0,0,'omitnan');
    m1 = mean(x1,'omitnan'); s1 = std(x1,0,'omitnan');

    % Normality: accept if either passes
    norm0 = (adtest(x0)==0) || (lillietest(x0)==0);
    norm1 = (adtest(x1)==0) || (lillietest(x1)==0);

    if norm0 && norm1
        [~, p] = ttest2(x0, x1);
        testType = 't-test';
    else
        p = ranksum(x0, x1);
        testType = 'Mann-Whitney';
    end

    % |d|
    d_abs = (abs(m0 - m1) ./ max(eps, sqrt(((numel(x0)-1)*var(x0,0,'omitnan') + (numel(x1)-1)*var(x1,0,'omitnan')) ...
        / max(1, (numel(x0)+numel(x1)-2)))));

    % StatEffect category (your rules)
    if p >= alpha
        statEffect = "not-sig";
    elseif d_abs >= 0.8
        statEffect = "sig-large";
    elseif d_abs >= 0.5
        statEffect = "sig-medium";
    elseif d_abs >= 0.2
        statEffect = "sig-small";
    elseif d_abs > 0
        statEffect = "sig-tiny";
    else
        statEffect = "not-sig";
    end

    out(f,:) = {features{f}, m0, s0, m1, s1, norm0, norm1, testType, p, d_abs, (m1 - m0), statEffect};
end

T = cell2table(out, 'VariableNames', ...
    {'Feature','Mean_LOW','Std_LOW','Mean_HIGH','Std_HIGH','Normal_LOW','Normal_HIGH', ...
    'Test','p_value','Abs_d','MeanDiff_HIGH_minus_LOW','StatEffect'});
end

function [subjLow, subjHigh, subj_ids] = build_subject_phase(all_data_phase, phaseTag)
% Returns cell arrays per subject: subjLow{s}: [n_i x 43], subjHigh{s}: [n_j x 43]
% phaseTag only for warning messages
% Detect number of subjects by file markers if present; else infer by block/epoch pattern
% Rebuild 43-feature vectors the same way you built combined_*.
% -----
% For calibration structs:
%   features_STEW(1:25 base, 26:31 STEW)
%   features_HEAT(26:31 HEAT)
%   features_MATB(26:31 MATB)
% For experiment structs:
%   STEW_features / HEAT_features / MATB_features (same columns)
% -----

fields_cal = {'features_STEW','features_HEAT','features_MATB','true_label'};
fields_exp = {'STEW_features','HEAT_features','MATB_features','true_label','adapted_epochs'};

isCal = all(isfield(all_data_phase, fields_cal));
isExp = all(isfield(all_data_phase, fields_exp));

if ~(isCal || isExp)
    error('build_subject_phase: unexpected struct fields for %s.', phaseTag);
end

% Infer subject IDs if field exists; else try to reconstruct by known counts
% We assume the original order concatenated subjects. If you stored subject indices, use them.
% Here we try to derive "subject change" by monotonic increases in a per-subject epoch counter if present.
% If not present, ask user to provide per-subject boundaries. For now we fall back to a single pseudo-subject.
if isfield(all_data_phase, 'subject_id')
    subj_ids = unique([all_data_phase.subject_id]);
else
    % Fallback: assume equal length per subject if metadata was consistent
    % Try to guess based on known design (calib ~600 epochs/subject; exp ~300 pre-adapt)
    N = numel(all_data_phase);
    guessCal = 600;
    guessExp = 300;
    if isCal && mod(N, guessCal)==0
        n_subjects = N / guessCal;
        subj_ids = 1:n_subjects;
        splits = reshape(1:N, guessCal, n_subjects);
    elseif isExp && mod(N, guessExp)==0
        n_subjects = N / guessExp;
        subj_ids = 1:n_subjects;
        splits = reshape(1:N, guessExp, n_subjects);
    else
        warning('Could not infer subject boundaries robustly; treating as one subject.');
        subj_ids = 1;
        splits = (1:N)';
    end
end

% If 'splits' not defined above via subject_id, define here from subj_ids
if ~exist('splits','var')
    % Group indices by subject_id
    N = numel(all_data_phase);
    subj_ids = unique([all_data_phase.subject_id]);
    n_subjects = numel(subj_ids);
    splits = cell2mat(arrayfun(@(sid) find([all_data_phase.subject_id]==sid).', subj_ids, 'UniformOutput', false));
    splits = reshape(splits, [], n_subjects);
end

n_subjects = numel(subj_ids);
subjLow  = cell(n_subjects,1);
subjHigh = cell(n_subjects,1);

for s = 1:n_subjects
    idxs = splits(:,s);
    idxs = idxs(:);

    if isCal
        stew = cell2mat({all_data_phase(idxs).features_STEW}');
        heat = cell2mat({all_data_phase(idxs).features_HEAT}');
        matb = cell2mat({all_data_phase(idxs).features_MATB}');
        y    = cell2mat({all_data_phase(idxs).true_label}');
    else
        % pre-adaptation only if adapted_epochs exists → filter
        if isfield(all_data_phase, 'adapted_epochs')
            keep = isnan([all_data_phase(idxs).adapted_epochs]);
            idxs = idxs(keep);
        end
        stew = cell2mat({all_data_phase(idxs).STEW_features}');
        heat = cell2mat({all_data_phase(idxs).HEAT_features}');
        matb = cell2mat({all_data_phase(idxs).MATB_features}');
        y    = cell2mat({all_data_phase(idxs).true_label}');
    end

    base  = stew(:,1:25);
    sstew = stew(:,26:31);
    sheat = heat(:,26:31);
    smatb = matb(:,26:31);
    X43   = [base, sstew, sheat, smatb];

    subjLow{s}  = X43(y==0,:);
    subjHigh{s} = X43(y==1,:);
end
end


function d = signed_cohens_d(X, y)
nF=size(X,2); d=zeros(nF,1);
x0=X(y==0,:); x1=X(y==1,:);
m0=mean(x0,1,'omitnan'); m1=mean(x1,1,'omitnan');
v0=var(x0,0,1,'omitnan'); v1=var(x1,0,1,'omitnan');
n0=size(x0,1); n1=size(x1,1);
sp = sqrt(((n0-1).*v0 + (n1-1).*v1) ./ max(1,(n0+n1-2)));
d  = (m1 - m0) ./ max(sp, eps); % HIGH - LOW
d  = d(:);
end


function plot_paired_subject_means(subjLow, subjHigh, features, feat_idx, phase_name)
nS = numel(subjLow);
for k = 1:numel(feat_idx)
    f = feat_idx(k);
    m0 = nan(nS,1); m1 = nan(nS,1);
    for s=1:nS
        m0(s)=mean(subjLow{s}(:,f),'omitnan');
        m1(s)=mean(subjHigh{s}(:,f),'omitnan');
    end
    subplot(ceil(numel(feat_idx)/2),2,k);
    plot([ones(nS,1), 2*ones(nS,1)]',[m0 m1]','-','Color',[0.7 0.7 0.7]); hold on;
    scatter(ones(nS,1), m0, 20, 'filled');
    scatter(2*ones(nS,1), m1, 20, 'filled');
    xlim([0.5 2.5]); set(gca,'XTick',[1 2],'XTickLabel',{'LOW','HIGH'});
    title(sprintf('%s — %s', phase_name, features{f}));
    ylabel('Subject mean'); grid on; box on;
end
sgtitle(['Paired subject means — ', phase_name]);
end


% -------- helper (local) --------
function val = local_extract_num(pattern, txt)
tok = regexp(txt, pattern, 'tokens', 'once');
if isempty(tok)
    val = NaN;
else
    val = str2double(tok{1});
end
end

function blk = get_section(txt, startLabel, endLabel)
    % Matches a line like '---- startLabel ----' up to the next '---- endLabel ----'
    pat = sprintf('-+\\s*%s\\s*-+(?s)(.*?)(?:\\n-+\\s*%s\\s*-+|\\Z)', startLabel, endLabel);
    tok = regexp(txt, pat, 'tokens', 'once');
    if isempty(tok)
        blk = '';
    else
        blk = tok{1};
    end
end

% ===== helpers =====
function s = regtok(pat, txt)
    t = regexp(txt, pat, 'tokens','once'); if isempty(t), s = 'NaN'; else, s = t{1}; end
end

function blk = extract_block(txt, mdl)
    % Grab the section between the dashed headers for each model
    hdr = ['---------------------- ' mdl ' ----------------------'];
    ids = strfind(txt, hdr);
    if isempty(ids), blk = ''; return; end
    a = ids(1);
    % end = next dashed header or end of text
    nxt = regexp(txt(a+1:end), '----------------------\s*[A-Z]+','once');
    if isempty(nxt), b = length(txt); else, b = a + nxt; end
    blk = txt(a:b);
end

function [TP,FP,FN,TN] = cm_from_counts(H,L,pH,pL,prec,rec,acc)
% Solve integer TP,FP,FN,TN given:
%  TP+FN = H, TN+FP = L, TP+FP = pH, TN+FN = pL
% And match precision/recall/accuracy within tolerance.
    maxTP = min(H, pH); 
    % Targets in proportions
    tPrec = prec/100; tRec = rec/100; tAcc = acc/100;
    best = [0 0 0 0]; bestErr = Inf;
    for TPc = 0:maxTP
        FPc = pH - TPc;
        FNc = H  - TPc;
        TNc = L  - FPc;
        if any([FPc FNc TNc] < 0), continue; end

        Prec = TPc / max(1, (TPc+FPc));
        Rec  = TPc / max(1, H);
        Acc  = (TPc+TNc) / max(1, H+L);

        err = 0;
        if ~isnan(tPrec), err = err + (Prec - tPrec)^2; end
        if ~isnan(tRec),  err = err + (Rec  - tRec )^2; end
        if ~isnan(tAcc),  err = err + (Acc  - tAcc )^2; end

        if err < bestErr
            best = [TPc FPc FNc TNc];
            bestErr = err;
        end
    end
    TP = best(1); FP = best(2); FN = best(3); TN = best(4);
end

function print_cm(name, CM)
    TP = CM(1,1); FP = CM(1,2); FN = CM(2,1); TN = CM(2,2);
    P  = TP + FP; N  = TN + FN; T = P + N;
    prec = TP / max(1,P);
    rec  = TP / max(1,TP+FN);
    f1   = 2*prec*rec / max(1e-12, prec+rec);
    acc  = (TP+TN) / max(1,T);
    fprintf('\n=== %s — POOLED (micro) ===\n', name);
    fprintf('Confusion Matrix (High=positive):\n');
    fprintf('[TP FP; FN TN] = [%d %d; %d %d]\n', TP,FP,FN,TN);
    fprintf('Precision/Recall/F1/Acc = %.2f / %.2f / %.2f / %.2f %%\n', ...
        100*prec, 100*rec, 100*f1, 100*acc);
end

% -------- helper (local) --------
function plot_cm_single(Cpf, nameStr)
    % Cpf is [TP FP; FN TN] with High = positive
    TP = Cpf(1,1); FP = Cpf(1,2); FN = Cpf(2,1); TN = Cpf(2,2);

    % Prepare for confusionchart (rows=true, cols=pred): [TP FN; FP TN]
    Cplot = [ TP  FN ; FP  TN ];

    % Metrics
    prec = TP / max(1, TP + FP);
    rec  = TP / max(1, TP + FN);
    f1   = (2*prec*rec) / max(1e-12, prec + rec);
    acc  = (TP + TN) / max(1, sum(Cpf(:)));

    % Plot
    figure('Color','w');
    confusionchart(Cplot, {'High','Low'}, ...
        'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
    title(sprintf('%s (pooled) — Acc %.2f%%  Prec %.2f%%  Rec %.2f%%  F1 %.2f%%', ...
        nameStr, 100*acc, 100*prec, 100*rec, 100*f1));

    % Optional save:
    % saveas(gcf, sprintf('%s_pooled_confusion.png', lower(nameStr)));
end

% -------- helper --------
function plot_conf_chart(Cpf, class_labels, context_str)
    % Cpf is [TP FP; FN TN] with High = positive.
    % Convert to confusionchart layout: rows=true {High,Low}, cols=pred {High,Low}
    C = [ Cpf(1,1)  Cpf(2,1) ;   % [TP  FN]
          Cpf(1,2)  Cpf(2,2) ];  % [FP  TN]

    figure;
    conf_chart = confusionchart(C, class_labels);
    conf_chart.Title = ['Confusion Matrix - ', context_str];
    conf_chart.RowSummary = 'row-normalized';
    conf_chart.ColumnSummary = 'column-normalized';
    conf_chart.FontSize = 20;
    conf_chart.FontName = 'Arial';
    conf_chart.GridVisible = 'on';
end