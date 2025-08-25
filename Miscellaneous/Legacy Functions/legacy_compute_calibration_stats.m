function compute_calibration_stats(pre_file, post_file)
% Compare Pre vs Post Calibration Accuracy
% Applies correct calibration mapping logic, and performs statistical tests

fprintf('[INFO] Running Calibrated Accuracy Delta Analysis...\n');

% -------------------- Load Pre-Calibration --------------------
T_pre = readtable(pre_file, 'Sheet', 'Total');
T_pre = T_pre(~contains(T_pre.TARGET, 'Within'), :);
T_pre.SOURCE = string(T_pre.SOURCE);
T_pre.TARGET = string(T_pre.TARGET);

% Extract fields
get_model = @(s) strtrim(extractAfter(s, ' '));
get_config = @(s) strtrim(extractBefore(s, ' '));
T_pre.Model = get_model(T_pre.SOURCE);
T_pre.Config = get_config(T_pre.SOURCE);
T_pre.Key = T_pre.SOURCE + "→" + T_pre.TARGET;

% Calibration Mapping
map_model_to_calib = containers.Map( ...
    {'STANDARD','HYPER','NORM','HYPER NORM'}, ...
    {{'finetuned','HYPER: FALSE'}, ...
     {'finetuned','HYPER: TRUE'}, ...
     {'adapted','HYPER: FALSE'; 'finetuned_adapted','HYPER: FALSE'}, ...
     {'adapted','HYPER: TRUE'; 'finetuned_adapted','HYPER: TRUE'}});

% -------------------- Load Post-Calibration --------------------
config_tags = {'24','csp','24wCsp'};

T_post_all = table( ...
    strings(0,1), ... % Source
    strings(0,1), ... % Target
    strings(0,1), ... % Config
    strings(0,1), ... % CalibrationType
    strings(0,1), ... % HYPER
    zeros(0,1), ...   % Accuracy
    strings(0,1), ... % Key
    'VariableNames', {'Source','Target','Config','CalibrationType','HYPER','Accuracy','Key'});

for i = 1:length(config_tags)
    tag = config_tags{i};
    sheetname = ['CalibResults_' tag];
    try
        T = readtable(post_file, 'Sheet', sheetname, 'ReadRowNames', true);
    catch
        warning('Skipping missing sheet: %s\n', sheetname);
        continue;
    end

    rowNames = string(T.Properties.RowNames);
    colNames = string(T.Properties.VariableNames);

    for r = 1:height(T)
        for c = 1:width(T)
            val = T{r, c};
            if ismissing(val) || ~iscell(val) || isempty(val{1}), continue; end
            str = regexprep(val{1}, '\s+', ' ');  % Flatten text

            match = regexp(str, 'Cross:\s*([\d.]+)%', 'tokens', 'once');
            if isempty(match), continue; end
            acc = str2double(match{1});

            row_match = regexp(rowNames(r), '([A-Za-z0-9_]+) \(Hyper:\s*(TRUE|FALSE)\)', 'tokens', 'once');
            col_match = regexp(colNames(c), '^([A-Za-z0-9_]+)_(adapted|finetuned|finetuned_adapted)', 'tokens', 'once');
            if isempty(row_match) || isempty(col_match), continue; end

            source_ds = row_match{1};
            hyper_flag = upper(row_match{2});
            target_ds = col_match{1};
            calib_type = col_match{2};

            entry = table;

            if hyper_flag == "TRUE"
                modeltype = "HYPER";
            else
                modeltype = "STANDARD";
            end
            entry.Source = source_ds + " " + tag + " " + modeltype;

            if calib_type == "finetuned_adapted" || calib_type == "adapted"
                if hyper_flag == "TRUE"
                    modeltype = "HYPER NORM";
                else
                    modeltype = "NORM";
                end
                entry.Source = source_ds + " " + tag + " " + modeltype;
            end

            entry.Target = target_ds + " " + tag + " " + entry.Source.extractAfter(" " + tag + " ");
            entry.Config = string(tag);
            entry.CalibrationType = calib_type;
            entry.HYPER = "HYPER: " + hyper_flag;
            entry.Accuracy = acc;
            entry.Key = entry.Source + "→" + entry.Target;

            T_post_all = [T_post_all; entry];
        end
    end
end

% Checking
[common_keys, i1, i2] = intersect(T_pre.Key, T_post_all.Key);
fprintf('[DEBUG] Matching entries: %d of %d\n', length(common_keys), height(T_pre));

% -------------------- Match and Filter --------------------
summary = {};

for i = 1:height(T_pre)
    mtype = T_pre.Model{i};
    if ~isKey(map_model_to_calib, mtype), continue; end
    mapping = map_model_to_calib(mtype);
    for j = 1:size(mapping, 1)
        calib = mapping{j, 1};
        hyper = mapping{j, 2};
        key = T_pre.Key(i);
        mask = strcmp(T_post_all.Key, key) & ...
               strcmp(T_post_all.CalibrationType, calib) & ...
               strcmp(T_post_all.HYPER, hyper);
        if any(mask)
            post_acc = T_post_all.Accuracy(find(mask, 1));
            delta = post_acc - T_pre.ACCURACY(i);
            summary = [summary; ...
                {T_pre.SOURCE(i), T_pre.TARGET(i), calib, ...
                 T_pre.Config(i), T_pre.Model(i), ...
                 T_pre.ACCURACY(i), post_acc, delta}];
        end
    end
end

if isempty(summary)
    summary = cell(0,8);  % Make sure it's 0x8 not 0x0
end

summary = cell2table(summary, 'VariableNames', ...
    {'Source','Target','CalibrationType','Config','Model','PreAccuracy','PostAccuracy','Delta'});

% Categorical types
summary.Config  = categorical(string(summary.Config), {'24','csp','24wCsp'}, 'Ordinal', true);
summary.Model   = categorical(string(summary.Model), {'STANDARD','HYPER','NORM','HYPER NORM'}, 'Ordinal', true);
summary.CalibrationType = categorical(string(summary.CalibrationType), ...
    {'adapted','finetuned','finetuned_adapted'}, 'Ordinal', true);

% Save summary
writetable(summary, post_file, 'Sheet', 'Delta_Accuracy');
fprintf('[SAVED] Delta Accuracy to Excel sheet: Delta_Accuracy\n');

% -------------------- Summary Stats --------------------
% Force column vectors:
summary.Config = summary.Config(:);
summary.Model = summary.Model(:);
summary.CalibrationType = summary.CalibrationType(:);

grouped = groupsummary(summary, {'CalibrationType','Config','Model'}, ...
    {'mean','std','median'}, 'Delta');
grouped.Properties.VariableNames(end-2:end) = {'MeanDelta','StdDelta','MedianDelta'};
writetable(grouped, post_file, 'Sheet', 'Delta_Summary');

% -------------------- Wilcoxon --------------------
fprintf('\n[STATISTICS] Wilcoxon Signed-Rank Tests (Delta > 0)\n');
delta_stats = groupsummary(summary, {'CalibrationType','Config','Model'}, 'mean', 'Delta');
groups = unique(delta_stats(:,1:3));

stat_results = {};
for i = 1:height(groups)
    mask = summary.CalibrationType == groups.CalibrationType(i) & ...
           summary.Config == groups.Config(i) & ...
           summary.Model == groups.Model(i);
    deltas = summary.Delta(mask);
    if numel(deltas) < 3, continue; end
    [p,~,stats] = signrank(deltas, 0, 'tail', 'right');
    stat_results = [stat_results; {
        string(groups.CalibrationType(i)), ...
        string(groups.Config(i)), ...
        string(groups.Model(i)), ...
        mean(deltas), std(deltas), p, stats.signedrank
    }];
end

if isempty(stat_results)
    stat_results = cell(0,7);  % Ensure it's 0x7 shape for the table
end

T_stats = cell2table(stat_results, ...
    'VariableNames', {'CalibrationType','Config','Model','MeanDelta','StdDelta','pValue','SignedRank'});
writetable(T_stats, post_file, 'Sheet', 'Delta_Wilcoxon');

% -------------------- Correlation --------------------
if height(summary) >= 3
    [RHO, PVAL] = corr(summary.PreAccuracy, summary.Delta, 'Type', 'Spearman');
    fprintf('[CORRELATION] Spearman ρ = %.3f, p = %.4f\n', RHO, PVAL);
    T_corr = table(RHO, PVAL, 'VariableNames', {'SpearmanRho','pValue'});
    writetable(T_corr, post_file, 'Sheet', 'Delta_Correlation');

    figure('Name','Correlation: Pre vs Delta');
    scatter(summary.PreAccuracy, summary.Delta, 60, 'filled'); grid on;
    xlabel('Pre Accuracy'); ylabel('Δ Accuracy (Post - Pre)');
    title(sprintf('Spearman ρ = %.2f (p = %.4f)', RHO, PVAL));

    % -------------------- Plot Correlation --------------------
    figure('Name','Correlation: Pre vs Delta');
    scatter(summary.PreAccuracy, summary.Delta, 60, 'filled'); grid on;
    xlabel('Pre Accuracy'); ylabel('Δ Accuracy (Post - Pre)');
    title(sprintf('Spearman ρ = %.2f (p = %.4f)', RHO, PVAL));
else
    fprintf('[CORRELATION] Skipped Spearman correlation — not enough data.\n');
end


% -------------------- 2-Way ANOVA --------------------
fprintf('\n[STATISTICS] 2-Way ANOVA on Delta Accuracy...\n');
[p, tbl, stats] = anovan(summary.Delta, ...
    {summary.Config, summary.Model}, ...
    'model', 2, ...
    'varnames', {'Config','Model'}, ...
    'display', 'off');

SS_total = tbl{end,2};
eta_config = tbl{2,2}/SS_total;
eta_model  = tbl{3,2}/SS_total;
eta_inter  = tbl{4,2}/SS_total;

fprintf('Config p = %.4f | η² = %.3f\n', p(1), eta_config);
fprintf('Model  p = %.4f | η² = %.3f\n', p(2), eta_model);
fprintf('Interaction p = %.4f | η² = %.3f\n', p(3), eta_inter);

anova_summary = table("Delta_Accuracy", p(1), eta_config, ...
                                     p(2), eta_model, ...
                                     p(3), eta_inter, ...
    'VariableNames', {'Context','Config_p','Config_Eta2','Model_p','Model_Eta2','Interaction_p','Interaction_Eta2'});
writetable(anova_summary, post_file, 'Sheet', 'Delta_ANOVA');

% -------------------- Bar Plot Effect Sizes --------------------
figure('Name', 'η² Effect Sizes (Delta Accuracy)');
bar([eta_config, eta_model, eta_inter]);
set(gca, 'XTickLabel', {'Config', 'Model', 'Interaction'});
ylabel('η²'); title('Effect Sizes on Δ Accuracy'); grid on;
ylim([0 1]);

fprintf('[✓] Calibrated accuracy delta analysis complete.\n');
end
