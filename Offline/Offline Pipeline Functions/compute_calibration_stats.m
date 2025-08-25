function compute_calibration_stats(results_file, params, config_tags)
% Computes statistical results from post-calibration results file
% Includes WITHIN and CROSS contexts and outputs Basics, ANOVA, and Posthoc sheets

outfile = sprintf('AfterCalib_%dTotal_%dCalib_Stats.xlsx', ...
    params.total_samples, params.calib_samples);

sheet_prefix = 'CalibResults_';
all_rows = {};

% ----------------------------
% Parse accuracy from all config sheets
% ----------------------------
for i = 1:length(config_tags)
    tag = config_tags{i};
    sheetname = [sheet_prefix tag];

    try
        T = readtable(results_file, 'Sheet', sheetname, 'ReadRowNames', true);
    catch
        warning('[SKIP] Could not read sheet: %s', sheetname);
        continue;
    end

    rowNames = string(T.Properties.RowNames);
    colNames = string(T.Properties.VariableNames);

    for r = 1:height(T)
        row_label = rowNames(r);
        row_match = regexp(row_label, '([A-Za-z0-9_]+)\s*\(Hyper:\s*(TRUE|FALSE)\)', 'tokens', 'once');
        if isempty(row_match), continue; end
        source_ds = row_match{1};
        hyper_flag = row_match{2};

        for c = 1:width(T)
            col_label = colNames{c};
            col_match = regexp(col_label, '([A-Za-z0-9_]+)_(adapted|finetuned|finetuned_adapted)_$', 'tokens', 'once');
            if isempty(col_match), continue; end
            target_ds = col_match{1};
            calib_type = col_match{2};

            str = T{r, c}{1};
            if isempty(str), continue; end
            str = regexprep(str, '\s+', ' ');  % normalize whitespace

            % Refine CalibrationType
            if ismember(calib_type, {'adapted', 'finetuned_adapted'})
                if contains(str, "NoNewModel(DA)")
                    calib_type = "adapted";
                else
                    calib_type = "finetuned_adapted";
                end
            end

            % Determine model type
            if calib_type == "adapted" || calib_type == "finetuned_adapted"
                model_type = "HYPER NORM";
                if hyper_flag == "FALSE"
                    model_type = "NORM";
                end
            elseif calib_type == "finetuned"
                model_type = "HYPER";
                if hyper_flag == "FALSE"
                    model_type = "STANDARD";
                end
            end

            source = source_ds + " " + tag + " " + model_type;
            target = target_ds + " " + tag + " " + model_type;
            config = tag;

            % Add WITHIN
            src_match = regexp(str, 'Source:\s*([\d.]+)%', 'tokens', 'once');
            if ~isempty(src_match)
                acc_source = str2double(src_match{1});
                all_rows = [all_rows; {
                    source, "WITHIN", acc_source, model_type, calib_type, config, "Within"
                }];
            end

            % Add CROSS
            cross_match = regexp(str, 'Cross:\s*([\d.]+)%', 'tokens', 'once');
            if ~isempty(cross_match)
                acc_cross = str2double(cross_match{1});
                all_rows = [all_rows; {
                    source, target, acc_cross, model_type, calib_type, config, "Cross"
                }];
            end
        end
    end
end

% Convert to table
T = cell2table(all_rows, ...
    'VariableNames', {'SOURCE','TARGET','ACCURACY','Model','CalibrationType','Config','Context'});

% ----------------------------
% BASIC STATS
% ----------------------------
T_stats = groupsummary(T, {'Config','Model','Context'}, {'mean','std'}, 'ACCURACY');
T_stats.Properties.VariableNames{'mean_ACCURACY'} = 'MeanAccuracy';
T_stats.Properties.VariableNames{'std_ACCURACY'} = 'StdAccuracy';
T_stats = sortrows(T_stats, {'Config','Model','Context'});
writetable(T_stats, outfile, 'Sheet', 'Basics');

% ----------------------------
% ANOVA + POSTHOC for Within/Cross separately
% ----------------------------
contexts = {'Within', 'Cross'};
anova_results = {};
posthoc_rows = {};

for c = 1:numel(contexts)
    ctx = contexts{c};
    subset = T(strcmp(T.Context, ctx), :);
    if isempty(subset), continue; end

    cfg_var = categorical(subset.Config);
    mdl_var = categorical(subset.Model);

    fprintf('\n[INFO] Running 2-way ANOVA for context: %s\n', ctx);
    [p, tbl, stats] = anovan(subset.ACCURACY, ...
        {cfg_var, mdl_var}, ...
        'model', 2, ...
        'varnames', {'Config','Model'}, ...
        'display', 'off');

    % Effect sizes (η²)
    SS_total = tbl{end,2};
    eta_sq_cfg = tbl{2,2} / SS_total;
    eta_sq_mdl = tbl{3,2} / SS_total;
    eta_sq_inter = tbl{4,2} / SS_total;

    % Print summary
    fprintf('[%s ANOVA] p(Config)=%.4f, η²=%.3f | p(Model)=%.4f, η²=%.3f | p(Int)=%.4f, η²=%.3f\n', ...
        ctx, p(1), eta_sq_cfg, p(2), eta_sq_mdl, p(3), eta_sq_inter);

    % Save to results
    anova_results(end+1,:) = {ctx, p(1), eta_sq_cfg, p(2), eta_sq_mdl, p(3), eta_sq_inter};

    % POSTHOC tests
    if p(1) < 0.05

        % Show plot
        figure('Name', sprintf('Post-hoc Model (%s)', ctx));
        multcompare(stats, 'Dimension', 1, 'Display', 'on');
        title(['Post-hoc: Model (' ctx ')']);

        comp = multcompare(stats, 'Dimension', 1, 'Display', 'off');
        labels = categories(cfg_var);
        for j = 1:size(comp,1)
            posthoc_rows(end+1,:) = {ctx, 'Config', labels{comp(j,1)}, labels{comp(j,2)}, comp(j,6), comp(j,6)<0.05};
        end
    end
    if p(2) < 0.05

        % Show plot
        figure('Name', sprintf('Post-hoc Model (%s)', ctx));
        multcompare(stats, 'Dimension', 2, 'Display', 'on');
        title(['Post-hoc: Model (' ctx ')']);

        comp = multcompare(stats, 'Dimension', 2, 'Display', 'off');
        labels = categories(mdl_var);
        for j = 1:size(comp,1)
            posthoc_rows(end+1,:) = {ctx, 'Model', labels{comp(j,1)}, labels{comp(j,2)}, comp(j,6), comp(j,6)<0.05};
        end
    end

    % Optional: effect size plot
    figure('Name', sprintf('Effect Sizes (%s)', ctx), 'NumberTitle', 'off');
    bar([eta_sq_cfg, eta_sq_mdl, eta_sq_inter]);
    set(gca, 'XTickLabel', {'Config', 'Model', 'Interaction'});
    ylabel('η² (Effect Size)');
    title(['Effect Sizes for ' ctx ' Context']);
    ylim([0 1]); grid on;
end

% ----------------------------
% Write ANOVA
% ----------------------------
tooltipRow = {
    'Type of analysis (Within/Cross)', ...
    'p-value for Config', 'η² effect size for Config', ...
    'p-value for Model', 'η² effect size for Model', ...
    'p-value for Interaction', 'η² effect size for Interaction'
};
headers = {
    'Context', ...
    'Config_p', 'Config_Eta2', ...
    'Model_p', 'Model_Eta2', ...
    'Interaction_p', 'Interaction_Eta2'
};

anovaT = cell2table(anova_results, 'VariableNames', headers);
writecell(tooltipRow, outfile, 'Sheet', 'ANOVA', 'Range', 'A1');
writetable(anovaT, outfile, 'Sheet', 'ANOVA', 'Range', 'A2');

% ----------------------------
% Write Posthoc (if any)
% ----------------------------
if isempty(posthoc_rows)
    writecell({'No significant ANOVA effects found → Posthoc skipped.'}, ...
        outfile, 'Sheet', 'Posthoc', 'Range', 'A1');
else
    posthocT = cell2table(posthoc_rows, ...
        'VariableNames', {'Context','EffectType','GroupA','GroupB','p_value','IsSignificant'});
    tooltip = {
        'Context', 'Effect Type', ...
        'Group A', 'Group B', ...
        'p-value', 'Is p < 0.05?'
    };
    writecell(tooltip, outfile, 'Sheet', 'Posthoc', 'Range', 'A1');
    writetable(posthocT, outfile, 'Sheet', 'Posthoc', 'Range', 'A2');
end

fprintf('Calibration Stats saved to: %s\n', outfile);
end
