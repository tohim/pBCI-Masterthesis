%% Compute Order Effect not by averaging features, but by collecting all features for each order pair and keeping individual subjects as a parameter
% ==> REPEATED-MEASURES ANOVA (includes the different subjects as a
% variable / does not assume that all features x order_type pairs are
% independent - as they actually come from different subjects -> therefore
% taking into account potential subject-specific variability in the EEG and
% the features)


% Calibration phase (only use last 8)
seq_calib_total = [1 1 0 0 0 1 1 0 1 0 1 0 0 1 1 1 1 0 0 0];
seq_calib_used = seq_calib_total(end-7:end);  % 01111000

% Experiment phase
seq_exp = [1 0 1 0 1 1 1 0 0 0];

% Define helper to get order type
get_order_type = @(a, b) sprintf('%d%d', a, b);

% Calib Total
calib_total_pairs = [seq_calib_total(1:end-1)' seq_calib_total(2:end)'];    % All contained block labels in position "i and i+1" 
calib_total_types = arrayfun(@(i) get_order_type(calib_total_pairs(i,1), calib_total_pairs(i,2)), ...
                             1:size(calib_total_pairs,1), 'UniformOutput', false);  % All contained block pairs within the condition
% Calib Used
calib_used_pairs = [seq_calib_used(1:end-1)' seq_calib_used(2:end)'];
calib_used_types = arrayfun(@(i) get_order_type(calib_used_pairs(i,1), calib_used_pairs(i,2)), ...
                            1:size(calib_used_pairs,1), 'UniformOutput', false);
% Experiment
exp_pairs = [seq_exp(1:end-1)' seq_exp(2:end)'];
exp_types = arrayfun(@(i) get_order_type(exp_pairs(i,1), exp_pairs(i,2)), ...
                     1:size(exp_pairs,1), 'UniformOutput', false);


n_features = 43;
blocks_per_subject = 10;
epochs_per_block   = 19;
epochs_per_subject = blocks_per_subject * epochs_per_block;


% (1 & 2) Total and Used Calibration Features per Block

blocks_calib_per_subject = 20;

% Split concatenated calibration logs back into per-subject cells
calib_per_subj = cell(n_subjects,1);
calib_len_per_subj = numel(all_data_calib)/n_subjects;  % should be 600 epochs per subject if 30 epochs/block
for s = 1:n_subjects
    from = (s-1)*calib_len_per_subj + 1;
    to   = s*calib_len_per_subj;
    calib_per_subj{s} = all_data_calib(from:to);
end

% Make block means per subject (20×43) then slice last 8 blocks as "used"
features_calib_total = cell(n_subjects,1);   % each: [20×43]
features_calib_used  = cell(n_subjects,1);   % each: [ 8×43] (last 8)

for s = 1:n_subjects
    C = calib_per_subj{s};
    blocks = unique([C.block]);              % expect 1:20
    Bt = zeros(numel(blocks), n_features);
    for b = 1:numel(blocks)
        idx = find([C.block] == blocks(b));
        Xb  = arrayfun(@(k) calib_combine43(C(idx(k))), 1:numel(idx), 'uni', false);
        Xb  = cat(1, Xb{:});                 % [epochs_in_block × 43], likely 30×43
        Bt(b,:) = mean(Xb,1);
    end
    features_calib_total{s} = Bt;            % 20×43
    features_calib_used{s}  = Bt(end-7:end,:); % last 8 blocks → 8×43
end


% (3) Experiment Features per Block 
% Pre-ADAPT epochs only
pred_idxs     = isnan([all_data_exp.adapted_epochs]);
pre_adapt_exp = all_data_exp(pred_idxs);

% features_exp{s} is [10×43] for subject s (block means)
features_exp = cell(n_subjects,1);
for s = 1:n_subjects
    subj = pre_adapt_exp((s-1)*epochs_per_subject + (1:epochs_per_subject));
    B = zeros(blocks_per_subject, n_features);
    for b = 1:blocks_per_subject
        idx = (b-1)*epochs_per_block + (1:epochs_per_block);
        Xb  = arrayfun(@(k) exp_combine43(subj(idx(k))), 1:epochs_per_block, 'uni', false);
        Xb  = cat(1, Xb{:});                  % [19×43]
        B(b,:) = mean(Xb,1);                  % block mean
    end
    features_exp{s} = B;                      % [10×43]
end



% Convert blocks → block-pairs per subject (means over two consecutive blocks)
% this is basically the "unit" of analysis that we are interested in (the
% block pairs)
% using to_pairs -> creates pair-level feature values averaging 2
% neighboring blocks. E.g.: pair 3 = mean(block 3, block 4)
% -> gives 1 value per order type instance per subject 

% Helper: from [B×F] block means to [P×F] pair means (B-1 pairs)
to_pairs = @(M) 0.5*( M(1:end-1,:) + M(2:end,:) );  % average consecutive rows

% Experiment: 10→9 pairs per subject
pairs_exp  = cellfun(to_pairs, features_exp, 'uni', false);           % each [9×43]
% Calibration total: 20→19 pairs per subject
pairs_ctot = cellfun(to_pairs, features_calib_total, 'uni', false);   % each [19×43]
% Calibration used: 8→7 pairs per subject
pairs_cuse = cellfun(to_pairs, features_calib_used,  'uni', false);   % each [7×43]


% Function "rm_anova_order": Build long tables (Subject, OrderType, FeatureValue) and run mixed models
% Run over all 3 phases

% Phase: Experiment
p_exp  = rm_anova_order(pairs_exp,  exp_types,         n_subjects, n_features);

% Phase: Calibration-Total
p_ctot = rm_anova_order(pairs_ctot, calib_total_types, n_subjects, n_features);

% Phase: Calibration-Used
p_cuse = rm_anova_order(pairs_cuse, calib_used_types,  n_subjects, n_features);


% Bonferroni Correction
alpha_bonf = 0.05 / n_features;

sig_bonf_calib_total = find(p_ctot < alpha_bonf);
sig_bonf_calib_used  = find(p_cuse  < alpha_bonf);
sig_bonf_exp   = find(p_exp   < alpha_bonf);

disp('--- Repeated-measures ANOVA (OrderType), Bonferroni corrected ---');
fprintf('[Calibration-Total] significant features: %s\n', mat2str(sig_bonf_calib_total));
fprintf('[Calibration-Used]  significant features: %s\n', mat2str(sig_bonf_calib_used));
fprintf('[Experiment]        significant features: %s\n', mat2str(sig_bonf_exp));

% Holm–Bonferroni across 43 features (per phase)
sig_exp  = holm_bonferroni(p_exp);
sig_ctot = holm_bonferroni(p_ctot);
sig_cuse = holm_bonferroni(p_cuse);

fprintf('\n--- Repeated-measures ANOVA (OrderType), Holm-corrected ---\n');
fprintf('[Calibration-Total]  significant features: %s\n', mat2str(find(sig_ctot)));
fprintf('[Calibration-Used]   significant features: %s\n', mat2str(find(sig_cuse)));
fprintf('[Experiment]         significant features: %s\n', mat2str(find(sig_exp)));



%% Check for consistency within data

% Look at first row of features_exp (subject 1)
% then take the very first 1x1 valu: should be feature 1: mean over 19
% epochs for block 1 (within subject 1)

% Load experiment_log of subject 1 (Results)

mean_total = [];

for i = 1:19
    s1_b1_mean = mean(experiment_log(i).STEW_features(1));
    mean_total = [mean_total, s1_b1_mean];
end

mean_final = mean(mean_total);

disp(mean_final);

% --> correct.


%% HELPERS
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


