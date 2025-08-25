%% NASA-TLX Analysis (Calibration vs Experiment; Balancing Check)
% Classical, readable version:
% - Primary: paired t-test on mean difference (report mean diff, 95% CI, Cohen's dz)
% - Sensitivity: Wilcoxon signed-rank p
% - Advisory: Shapiro–Wilk/Francia (via swtest.m) on paired differences
% - Per-subscale analyses included
%
% Requires: Statistics and Machine Learning Toolbox
% Also requires: swtest.m in the same folder (Shapiro–Wilk/Francia)

clear; close all; clc;

set(groot,'defaultAxesTickLabelInterpreter','none');
set(groot,'defaultTextInterpreter','none');
set(groot,'defaultLegendInterpreter','none');


%% ---- 1) Load data ----
excelPath = 'E:\SchuleJobAusbildung\HTW\MasterThesis\Code\FiguresTablesLegacy\NASA-TLX Results.xlsx';  % <<<< set if needed
T = readtable(excelPath, 'Sheet', 'Tabelle1');

% Expected subscales
subscales = ["MentalDemand","PhysicalDemand","TemporalDemand","Performance","Effort","Frustration"];
assert(all(ismember(subscales, string(T.Properties.VariableNames))), ...
    'Missing one or more NASA-TLX subscale columns.');

% Normalize text columns
T.Phase = string(T.Phase);
T.Condition = string(T.Condition);
T.PhaseCond = T.Phase + "_" + T.Condition;

% Reverse Performance and compute overall TLX (higher = more workload)
T.Performance_rev = 100 - T.Performance;
sixCols = ["MentalDemand","PhysicalDemand","TemporalDemand","Performance_rev","Effort","Frustration"];
T.TLX_overall = mean(T{:, sixCols}, 2, 'omitnan');

%% ---- 2) Wide table (overall TLX) ----
S = T(:, ["Subject","PhaseCond","TLX_overall"]);
W = unstack(S, 'TLX_overall', 'PhaseCond', 'AggregationFunction', @mean);

expected = ["Calibration_LOW","Calibration_HIGH","Experiment_LOW","Experiment_HIGH"];
present  = expected(ismember(expected, string(W.Properties.VariableNames)));

keepVars = ["Subject", present];
Complete = rmmissing(W(:, keepVars), 'DataVariables', keepVars(2:end));

fprintf('Subjects with complete data for %s: N=%d\n', strjoin(present, ', '), height(Complete));

%% ---- 3) Descriptives (mean, SD, N, 95%% CI) ----
Desc = table('Size',[0 6], ...
    'VariableTypes', {'string','double','double','double','double','double'}, ...
    'VariableNames', {'PhaseCond','N','Mean','SD','CI95_low','CI95_high'});

for i = 1:numel(present)
    col = present(i);
    x = Complete.(col);
    [m, sd, n, ci] = meanCI(x);
    Desc = [Desc; {col, n, m, sd, ci(1), ci(2)}];
end

disp('--- Descriptives (Overall TLX) ---');
disp(Desc);

%% ---- 4) Paired comparisons (overall TLX) ----
Results = emptyResultsTable();

comparisons = { ...
    "Calibration: HIGH - LOW",      "Calibration_HIGH","Calibration_LOW"; ...
    "Experiment: HIGH - LOW",       "Experiment_HIGH","Experiment_LOW"; ...
    "Adaptation (LOW): Exp_LOW - Calib_LOW", "Experiment_LOW","Calibration_LOW"; ...
    "Adaptation (HIGH): Exp_HIGH - Calib_HIGH","Experiment_HIGH","Calibration_HIGH" ...
    };

for i = 1:size(comparisons,1)
    name = comparisons{i,1};
    a = comparisons{i,2}; b = comparisons{i,3};
    if all(ismember([a b], string(Complete.Properties.VariableNames)))
        x = Complete.(a); y = Complete.(b);
        R = pairedCompare_Classical(x, y, name);   % <<< stores t, CI, dz, Wilcoxon, SW
        Results = [Results; struct2table(R)];
    end
end

% Balancing: (Calib H-L) vs (Exp H-L)
need = ["Calibration_HIGH","Calibration_LOW","Experiment_HIGH","Experiment_LOW"];
if all(ismember(need, string(Complete.Properties.VariableNames)))
    calibDiff = Complete.("Calibration_HIGH") - Complete.("Calibration_LOW");
    expDiff   = Complete.("Experiment_HIGH") - Complete.("Experiment_LOW");
    R = pairedCompare_Classical(calibDiff, expDiff, "Balancing: (Calib H-L) - (Exp H-L)");
    Results = [Results; struct2table(R)];
end

disp('--- Paired Tests (Overall TLX) ---');
disp(Results);

%% ---- 5) Per-subscale paired analyses ----
SubResults = emptyResultsTable();
for s = sixCols
    S2 = T(:, ["Subject","PhaseCond", s]);
    W2 = unstack(S2, s, 'PhaseCond', 'AggregationFunction', @mean);

    present2 = expected(ismember(expected, string(W2.Properties.VariableNames)));
    keep2 = ["Subject", present2];
    C2 = rmmissing(W2(:, keep2), 'DataVariables', keep2(2:end));

    cset = { ...
        "Calibration: HIGH - LOW",      "Calibration_HIGH","Calibration_LOW"; ...
        "Experiment: HIGH - LOW",       "Experiment_HIGH","Experiment_LOW"; ...
        "Adaptation (LOW): Exp_LOW - Calib_LOW", "Experiment_LOW","Calibration_LOW"; ...
        "Adaptation (HIGH): Exp_HIGH - Calib_HIGH","Experiment_HIGH","Calibration_HIGH" ...
        };
    for i = 1:size(cset,1)
        name = s + " | " + cset{i,1};
        a = cset{i,2}; b = cset{i,3};
        if all(ismember([a b], string(C2.Properties.VariableNames)))
            x = C2.(a); y = C2.(b);
            R = pairedCompare_Classical(x, y, name);   % <<< same storage
            SubResults = [SubResults; struct2table(R)];
        end
    end

    if all(ismember(need, string(C2.Properties.VariableNames)))
        calibDiff = C2.("Calibration_HIGH") - C2.("Calibration_LOW");
        expDiff   = C2.("Experiment_HIGH") - C2.("Experiment_LOW");
        name = s + " | Balancing: (Calib H-L) - (Exp H-L)";
        R = pairedCompare_Classical(calibDiff, expDiff, name);
        SubResults = [SubResults; struct2table(R)];
    end
end

disp('--- Paired Tests (Per Subscale) ---');
disp(SubResults);

%% ---- 6) Save outputs ----
writetable(Desc, 'nasa_tlx_descriptives.csv');
writetable(Results, 'nasa_tlx_tests.csv');            % includes p_ttest, t_stat, CI, dz, p_wilcoxon, p_sw, W_sw
writetable(SubResults, 'nasa_tlx_subscale_tests.csv');
writetable(Complete, 'nasa_tlx_wide.csv');

%% ---- 7) Plots ----
% Boxplot of overall TLX by PhaseCond (all rows)
figure('Name','TLX Overall by Phase & Condition');
boxplot(T.TLX_overall, T.PhaseCond, 'Notch','off', 'Whisker',1.5);
ylabel('TLX (0–100)'); title('NASA-TLX Overall by Phase & Condition');
xtickangle(15);
saveas(gcf, 'tlx_overall_boxplot.png');

% Bar with 95% CI from descriptives (per-subject means)
figure('Name','Mean TLX with 95% CI');
labels = cellstr(Desc.PhaseCond);
x = 1:height(Desc);
bar(x, Desc.Mean);
hold on;
errLow  = Desc.Mean - Desc.CI95_low;
errHigh = Desc.CI95_high - Desc.Mean;
errorbar(x, Desc.Mean, errLow, errHigh, 'k', 'LineStyle','none', 'CapSize',8);
hold off;
xlim([0.5, height(Desc)+0.5]);
set(gca, 'XTick', x, 'XTickLabel', labels);
xtickangle(15);
ylabel('TLX (0–100)'); title('Mean NASA-TLX Overall (95% CI)');
saveas(gcf, 'tlx_overall_bar_ci.png');

disp('Saved: nasa_tlx_descriptives.csv, nasa_tlx_tests.csv, nasa_tlx_subscale_tests.csv, nasa_tlx_wide.csv');
disp('Saved: tlx_overall_boxplot.png, tlx_overall_bar_ci.png');


%% 7b) Paired spaghetti plots (overall TLX)
plotSpaghettiPair(Complete, "Calibration_LOW",  "Calibration_HIGH",  ...
    "Calibration LOW ↔ HIGH", "TLX (0–100)", "spaghetti_calib_LOW_HIGH.png");

plotSpaghettiPair(Complete, "Experiment_LOW",   "Experiment_HIGH",   ...
    "Experiment LOW ↔ HIGH", "TLX (0–100)", "spaghetti_experiment_LOW_HIGH.png");

plotSpaghettiPair(Complete, "Calibration_LOW",  "Experiment_LOW",    ...
    "LOW (Calibration ↔ Experiment)", "TLX (0–100)", "spaghetti_LOW_calib_exp.png");

plotSpaghettiPair(Complete, "Calibration_HIGH", "Experiment_HIGH",   ...
    "HIGH (Calibration ↔ Experiment)", "TLX (0–100)", "spaghetti_HIGH_calib_exp.png");

%% 7c) Paired-difference dot plots (overall TLX)
% These map directly to your tests:

% % HIGH − LOW (Calibration)
% plotDiffDots(Complete, "Calibration_HIGH", "Calibration_LOW", ...
%     "Difference: Calibration (HIGH − LOW)", "Difference in TLX (points)", "diff_calib_HminusL.png");
% 
% % HIGH − LOW (Experiment)
% plotDiffDots(Complete, "Experiment_HIGH", "Experiment_LOW", ...
%     "Difference: Experiment (HIGH − LOW)", "Difference in TLX (points)", "diff_experiment_HminusL.png");
% 
% % Adaptation (LOW): Experiment − Calibration (LOW)
% plotDiffDots(Complete, "Experiment_LOW", "Calibration_LOW", ...
%     "Difference: Adaptation (LOW) = Experiment − Calibration", "Difference in TLX (points)", "diff_adapt_LOW_ExpMinusCalib.png");
% 
% % Adaptation (HIGH): Experiment − Calibration (HIGH)
% plotDiffDots(Complete, "Experiment_HIGH", "Calibration_HIGH", ...
%     "Difference: Adaptation (HIGH) = Experiment − Calibration", "Difference in TLX (points)", "diff_adapt_HIGH_ExpMinusCalib.png");
% 
% % Balancing: (Calib H−L) − (Exp H−L)
% if all(ismember(["Calibration_HIGH","Calibration_LOW","Experiment_HIGH","Experiment_LOW"], string(Complete.Properties.VariableNames)))
%     calibDiff = Complete.("Calibration_HIGH") - Complete.("Calibration_LOW");
%     expDiff   = Complete.("Experiment_HIGH") - Complete.("Experiment_LOW");
%     d_bal = calibDiff - expDiff;
%     plotDiffDotsVector(d_bal, "Difference: Balancing (Calib gap − Exp gap)", ...
%         "Difference in TLX (points)", "diff_balancing_gap.png", "Balancing: (Calib H−L) − (Exp H−L)");
% end



%% Correlate NASA TLX results with Features

% Collecting all data

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

% CALIBRATION DATA

true_label = cell2mat({all_data_calib.true_label})';

low_idx = true_label == 0;
high_idx = true_label == 1;

% Convert features from cell to matrix
all_epoch_features_STEW_calib = cell2mat({all_data_calib.features_STEW}');
all_epoch_features_HEAT_calib = cell2mat({all_data_calib.features_HEAT}');
all_epoch_features_MATB_calib = cell2mat({all_data_calib.features_MATB}');

base_low_calib = all_epoch_features_STEW_calib(low_idx, 1:25);
base_high_calib = all_epoch_features_STEW_calib(high_idx, 1:25);

stew_low_calib = all_epoch_features_STEW_calib(low_idx, 26:31);
stew_high_calib = all_epoch_features_STEW_calib(high_idx, 26:31);

heat_low_calib = all_epoch_features_HEAT_calib(low_idx, 26:31);
heat_high_calib = all_epoch_features_HEAT_calib(high_idx, 26:31);

matb_low_calib = all_epoch_features_MATB_calib(low_idx, 26:31);
matb_high_calib = all_epoch_features_MATB_calib(high_idx, 26:31);

combined_low_calib = [base_low_calib, stew_low_calib, heat_low_calib, matb_low_calib];
combined_high_calib = [base_high_calib, stew_high_calib, heat_high_calib, matb_high_calib];


% EXPERIMENT 

% Only look at predictions vs true labels before applying "ADAPT"
pred_idxs = isnan([all_data_exp.adapted_epochs]);

% Convert features from cell to matrix
all_epoch_features_STEW_exp = cell2mat({all_data_exp(pred_idxs).STEW_features}');
all_epoch_features_HEAT_exp = cell2mat({all_data_exp(pred_idxs).HEAT_features}');
all_epoch_features_MATB_exp = cell2mat({all_data_exp(pred_idxs).MATB_features}');

true_label = cell2mat({all_data_exp(pred_idxs).true_label})';

low_idx = true_label == 0;
high_idx = true_label == 1;

base_low_exp = all_epoch_features_STEW_exp(low_idx, 1:25);
base_high_exp = all_epoch_features_STEW_exp(high_idx, 1:25);

stew_low_exp = all_epoch_features_STEW_exp(low_idx, 26:31);
stew_high_exp = all_epoch_features_STEW_exp(high_idx, 26:31);

heat_low_exp = all_epoch_features_HEAT_exp(low_idx, 26:31);
heat_high_exp = all_epoch_features_HEAT_exp(high_idx, 26:31);

matb_low_exp = all_epoch_features_MATB_exp(low_idx, 26:31);
matb_high_exp = all_epoch_features_MATB_exp(high_idx, 26:31);

combined_low_exp = [base_low_exp, stew_low_exp, heat_low_exp, matb_low_exp];
combined_high_exp = [base_high_exp, stew_high_exp, heat_high_exp, matb_high_exp];


%%

feat43 = [ ...
 {'Theta Power','Alpha Power','Beta Power', ...
  'Theta Alpha Ratio','Theta Beta Ratio','Engagement Index', ...
  'Theta Frontal','Theta Temporal','Theta Parietal','Theta Occipital', ...
  'Alpha Frontal','Alpha Temporal','Alpha Parietal','Alpha Occipital', ...
  'Beta Frontal','Beta Temporal','Beta Parietal', ...
  'Avg Coherence','Theta Coherence','Alpha Coherence', ...
  'Avg Mobility','Avg Complexity','Avg Entropy', ...
  'Theta Entropy','Alpha Entropy'}, ...
 arrayfun(@(k) sprintf('CSP%d LOW STEW',k),1:3,'uni',0), ...
 arrayfun(@(k) sprintf('CSP%d HIGH STEW',k),1:3,'uni',0), ...
 arrayfun(@(k) sprintf('CSP%d LOW HEAT',k),1:3,'uni',0), ...
 arrayfun(@(k) sprintf('CSP%d HIGH HEAT',k),1:3,'uni',0), ...
 arrayfun(@(k) sprintf('CSP%d LOW MATB',k),1:3,'uni',0), ...
 arrayfun(@(k) sprintf('CSP%d HIGH MATB',k),1:3,'uni',0) ...
];
V = matlab.lang.makeValidName(string(feat43));   % valid table var names

% ----- CALIBRATION -----
epochs_per_subject_calib = 600;
subj_id_calib = repelem(kept_subjects(:), epochs_per_subject_calib);      % 5400×1

true_label_calib = cell2mat({all_data_calib.true_label}');                % 0/1
X43_calib = [ all_epoch_features_STEW_calib(:,1:25), ...
              all_epoch_features_STEW_calib(:,26:31), ...
              all_epoch_features_HEAT_calib(:,26:31), ...
              all_epoch_features_MATB_calib(:,26:31) ];

[Gcal, S_cal, L_cal] = findgroups(subj_id_calib, true_label_calib);
Xmean_cal = splitapply(@mean, X43_calib, Gcal);

Cond_cal = repmat("LOW", numel(L_cal),1);  Cond_cal(L_cal==1)="HIGH";
Tcal = table(S_cal, repmat("Calibration",numel(S_cal),1), Cond_cal, ...
             array2table(Xmean_cal, 'VariableNames', cellstr(V)), ...
             'VariableNames', {'Subject','Phase','Condition','Features'});
Tcal = [Tcal(:,1:3) Tcal.Features];   % flatten

% ----- EXPERIMENT (pre-adapt only, using your pred_idxs) -----
epochs_per_subject_exp = 300;
subj_id_exp = repelem(kept_subjects(:), epochs_per_subject_exp);          % 2700×1
subj_id_exp = subj_id_exp(pred_idxs);                                     % filter to pre-adapt

true_label_exp = cell2mat({all_data_exp(pred_idxs).true_label}');         % 0/1
X43_exp = [ all_epoch_features_STEW_exp(:,1:25), ...
            all_epoch_features_STEW_exp(:,26:31), ...
            all_epoch_features_HEAT_exp(:,26:31), ...
            all_epoch_features_MATB_exp(:,26:31) ];

[Gexp, S_exp, L_exp] = findgroups(subj_id_exp, true_label_exp);
Xmean_exp = splitapply(@mean, X43_exp, Gexp);

Cond_exp = repmat("LOW", numel(L_exp),1);  Cond_exp(L_exp==1)="HIGH";
Texp = table(S_exp, repmat("Experiment",numel(S_exp),1), Cond_exp, ...
             array2table(Xmean_exp, 'VariableNames', cellstr(V)), ...
             'VariableNames', {'Subject','Phase','Condition','Features'});
Texp = [Texp(:,1:3) Texp.Features];

% Combine
TfeatAgg = [Tcal; Texp];                      % rows = Subject×Phase×Condition

% Harmonize key types
T.Subject   = double(T.Subject);
T.Phase     = string(T.Phase);        % 'Calibration' / 'Experiment'
T.Condition = string(T.Condition);    % 'LOW' / 'HIGH'

J = innerjoin(TfeatAgg, T, 'Keys', {'Subject','Phase','Condition'});
% Now J has: Subject, Phase, Condition, 43 feature columns, and TLX columns


y = J.MentalDemand;           % choose any TLX column here
X = J{:, cellstr(V)};         % all 43 features
Z = (X - mean(X,1,'omitnan')) ./ std(X,0,1,'omitnan');   % z-score columns

P = numel(V);
beta = zeros(P,1); se = zeros(P,1); pval = zeros(P,1);

for j = 1:P
    zj = Z(:,j); good = isfinite(zj) & isfinite(y);
    Xj = [ones(sum(good),1) zj(good)];
    bj = Xj \ y(good);
    r  = y(good) - Xj*bj;
    s2 = sum(r.^2)/(sum(good)-2);
    covb = s2*inv(Xj.'*Xj);
    beta(j) = bj(2);  se(j) = sqrt(covb(2,2));
    t = beta(j)/se(j);
    pval(j) = 2*tcdf(-abs(t), sum(good)-2);
end

ci95 = 1.96*se;
q = bhFDR(pval);                          % FDR across 43 tests

Res = table(string(feat43(:)), beta, ci95, pval, q, ...
            'VariableNames', {'Feature','Beta_TLX_per1SD','CI95','p','q_FDR'});
Res = sortrows(Res, -abs(Res.Beta_TLX_per1SD));
disp(Res(1:10,:));

% Plot top-20
K = min(20, numel(V)); Rk = Res(1:K,:); yidx = (1:K)'; sig = Rk.q_FDR < 0.05;
figure('Color','w','Units','centimeters','Position',[0 0 18 14]); ax=axes; hold on;
barh(ax, Rk.Beta_TLX_per1SD, 'FaceColor',[0.30 0.30 0.80]);
errorbar(Rk.Beta_TLX_per1SD, yidx, Rk.CI95, 'horizontal','k','LineStyle','none','CapSize',6,'LineWidth',0.9);
plot(Rk.Beta_TLX_per1SD(sig), yidx(sig), 'kd','MarkerFaceColor','k','MarkerSize',4);
yticks(yidx); yticklabels(Rk.Feature); set(ax,'YDir','reverse');
xline(0,'k-'); grid on; box off;
xlabel('Mental Demand change (points) per +1 SD in feature');
title('Associations with NASA-TLX Mental Demand (FDR-corrected)');




%% ===== Helper functions =====
function [m, sd, n, ci] = meanCI(x, alpha)
    if nargin < 2, alpha = 0.05; end
    x = x(:); x = x(~isnan(x));
    n = numel(x);
    m = mean(x); sd = std(x, 0);
    if n > 1
        se = sd/sqrt(n);
        tcrit = tinv(1 - alpha/2, n - 1);
        ci = [m - tcrit*se, m + tcrit*se];
    else
        ci = [NaN NaN];
    end
end

function Tbl = emptyResultsTable()
    % Columns include: p_ttest, t_stat, CI, dz, p_wilcoxon, p_sw, W_sw
    Tbl = table('Size',[0 13], ...
        'VariableTypes', {'string','double','double','double','double','double','double','double','double','double','double','double','double'}, ...
        'VariableNames', {'Comparison','N','MeanDiff','MedianDiff', ...
                          'p_ttest','t_stat','CI95_low','CI95_high','dz', ...
                          'p_wilcoxon','p_sw','W_sw','H_sw'});
end

function R = pairedCompare_Classical(x, y, label)
    % Primary: paired t-test; also returns Wilcoxon and Shapiro–Wilk/Francia on diffs
    x = x(:); y = y(:);
    d = x - y; d = d(~isnan(d));
    n = numel(d);

    R = struct('Comparison', string(label), 'N', n, ...
        'MeanDiff', NaN, 'MedianDiff', NaN, ...
        'p_ttest', NaN, 't_stat', NaN, 'CI95_low', NaN, 'CI95_high', NaN, 'dz', NaN, ...
        'p_wilcoxon', NaN, 'p_sw', NaN, 'W_sw', NaN, 'H_sw', NaN);

    if n == 0, return; end

    % Descriptives of paired difference
    R.MeanDiff   = mean(d, 'omitnan');
    R.MedianDiff = median(d, 'omitnan');

    % --- Paired t-test (primary classical result) ---
    [~, p_t, ci_t, stats_t] = ttest(x, y);         % paired by default
    R.p_ttest   = p_t;
    R.t_stat    = stats_t.tstat;
    R.CI95_low  = ci_t(1);
    R.CI95_high = ci_t(2);

    % Cohen's dz (paired)
    sdd = std(d, 0);
    if sdd > 0, R.dz = mean(d)/sdd; end

    % --- Wilcoxon signed-rank (sensitivity) ---
    try
        [p_w, ~, ~] = signrank(x, y);
        R.p_wilcoxon = p_w;
    catch
        R.p_wilcoxon = NaN;
    end

    % --- Shapiro–Wilk/Francia (advisory) on paired differences ---
    % (Requires swtest.m in path; may auto-switch to SF for leptokurtic data)
    try
        [Hsw, p_sw, W_sw] = swtest(d, 0.05);   % your function
        R.p_sw = p_sw; R.W_sw = W_sw; R.H_sw = Hsw;
    catch
        R.p_sw = NaN; R.W_sw = NaN; R.H_sw = NaN;
    end
end


function plotSpaghettiPair(Complete, varA, varB, titleStr, ylab, saveAs)
    % Simple paired spaghetti plot: one line per subject connecting varA -> varB
    A = Complete.(varA); B = Complete.(varB);
    n = height(Complete);
    figure('Name', titleStr); hold on;
    for i = 1:n
        plot([1 2], [A(i) B(i)], '-o', 'LineWidth', 1);
    end
    % Overlay mean ± 95% CI for each side
    [mA, ~, ~, ciA] = meanCI(A);
    [mB, ~, ~, ciB] = meanCI(B);
    errorbar(1, mA, mA-ciA(1), ciA(2)-mA, 'k', 'LineWidth', 2, 'CapSize', 10);
    errorbar(2, mB, mB-ciB(1), ciB(2)-mB, 'k', 'LineWidth', 2, 'CapSize', 10);
    xlim([0.5 2.5]); set(gca, 'XTick', [1 2], 'XTickLabel', {char(varA), char(varB)});
    ylabel(ylab); title(titleStr); grid on; box off;
    if nargin >= 6 && ~isempty(saveAs), saveas(gcf, saveAs); end
end

function plotDiffDots(Complete, varNum, varDen, titleStr, ylab, saveAs)
    % Dot plot of paired differences with mean ± 95% CI (varNum - varDen)
    d = Complete.(varNum) - Complete.(varDen);
    plotDiffDotsVector(d, titleStr, ylab, saveAs, sprintf('%s − %s', char(varNum), char(varDen)));
end

function plotDiffDotsVector(d, titleStr, ylab, saveAs, subtitleStr)
    % Same as above but from a raw vector of differences d
    d = d(:); d = d(~isnan(d));
    n = numel(d);
    [m, ~, ~, ci] = meanCI(d);
    figure('Name', titleStr); hold on;
    % Jittered vertical dots at x=1
    x = 1 + (rand(n,1)-0.5)*0.12;
    plot(x, d, 'o', 'MarkerFaceColor', [0.3 0.3 0.3], 'MarkerEdgeColor', 'none');
    % Zero reference, mean, and CI
    plot([0.7 1.3], [0 0], ':', 'Color', [0.5 0.5 0.5]);
    plot([0.85 1.15], [m m], 'k-', 'LineWidth', 2);
    errorbar(1, m, m-ci(1), ci(2)-m, 'k', 'LineWidth', 2, 'CapSize', 12);
    xlim([0.6 1.4]);
    set(gca, 'XTick', 1, 'XTickLabel', "difference");
    ylabel(ylab); title({titleStr; subtitleStr}); grid on; box off;
    if nargin >= 5 && ~isempty(saveAs), saveas(gcf, saveAs); end
end 


function q = bhFDR(p)
    % Benjamini–Hochberg FDR for a vector p (NaNs preserved)
    p = p(:);
    isn = isnan(p);
    ps  = p(~isn);
    [ps, ix] = sort(ps,'ascend');
    m = numel(ps);
    ranks = (1:m)';
    qvals = (ps .* m) ./ ranks;
    % enforce monotone nonincreasing from the end
    for k = m-1:-1:1
        qvals(k) = min(qvals(k), qvals(k+1));
    end
    q = nan(size(p));
    q(~isn) = qvals(invperm(ix));   % put back in original order
end
function ip = invperm(ix)
    ip = zeros(size(ix)); ip(ix) = 1:numel(ix);
end

