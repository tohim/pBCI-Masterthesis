function compute_stats(output_name, opts)

% -------------------------------------------------------------------------
% Readme:
%
% What the Data is:
% Each row in the summaryT table contains:
% Config → the feature configuration: '24', 'csp', '24wCsp'
% Model → the model type: 'STANDARD', 'HYPER', 'NORM', 'HYPER NORM'
% Context → either 'Within' or 'Cross', depending on if the source and target were the same
% Accuracy → the classification accuracy achieved with that config+model+context
%
% How the boxplots are generated:
% ctxT = summaryT(summaryT.Context == ctx, :);
% boxplot(ctxT.Accuracy, ctxT.Config);  % left subplot
% boxplot(ctxT.Accuracy, ctxT.Model);   % right subplot
%
% So the boxplots display the distribution of accuracies grouped by:
% Left subplot → Feature Config (for a fixed context)
% Right subplot → Model Type (for a fixed context)
%
% Each box in the plot shows:
% The median (red line)
% 25th and 75th percentiles (box edges)
% Whiskers = data range excluding outliers
% Red + = outliers

% 2-way ANOVA (Anaylsis of Variance) = statistical test to evalute the
% effect of 2 independent factors (here Config and Model) on a dependent
% variable (Accuracy). And it tests whether there is a statistical
% interaction between the 2 factors.
% Conventional Threshold for statistical significance = p < 0.05
% "There is a less than 5% probability that the observed differences are
% due to random chance".
% ANOVA here is done separately for Within and Cross contexts.

% ANOVA Output:
% p-Value for Config: Tells if different feature configurations lead to
% different accuracy distributions within a context
% p-Value for Model: Tells if different models lead to different accuracy
% distributions witin a context
% p-Value for Interaction: Tells if there is a significant interaction
% between Config & Model. (e.g. "Config A performs better only with Model B)

% Effect Size (η² / eta squared) tells if the observed statistical
% significant effect is also meaningful in practice.
% Using eta squared because ANOVA is a variance-based test across multiple
% groups:
% η² = proportion of the total variance in the dependent variable explained by a factor (e.g. Config or Model).
% eta²_config = SS_config / SS_total
% This tells us: "How much of the accuracy variation is explained by the config type?”
% We cannot use Cohen's d here because it is designed for comparing 2
% groups only (not valid for multi-level categorical predictors)

% Effect Size Output:
% 0.01 - 0.06 --> Small Effect
% 0.06 - 0.14 --> Medium Effect
% > 0.14 --> Large Effect

% Post-hoc analysis: Multiple Comparison Plots (Marginal Means) (using
% Statistic Toolbox)
% Each plot shows pairwise comparisons between levels of a factor (Config
% or Model), after a significant ANOVA result.
% Each horizontal line represents a group's mean accuracy with confidence intervals
% Overlapping intervals => groups not significantly different
% Non-Overlapping intervals => significant difference
% Bottom Message: explicitly states which comparisons are statistically significant
%
% Sheets Content: 
% Basics: Shows descriptive stats like mean/std of accuracies grouped by feature config and model type, 
% split by Within vs Cross. It gives an overview of which model/feature combo is robust on average.
% -------------------------------------------------------------------------

% Load classification results
T = readtable(output_name);

% Prepare arrays
N = height(T);
config_list = {};
model_list = {};
context_list = {};
accuracy_list = [];

for i = 1:N
    source = T.SOURCE{i};
    target = T.TARGET{i};
    acc = T.ACCURACY(i);
    if isnan(acc), continue; end

    % Detect context
    is_within = contains(target, 'Within', 'IgnoreCase', true);
    context = 'Within';
    if ~is_within, context = 'Cross'; end

    % Extract config and model
    tokens = strsplit(source, ' ');
    cfg = extract_cfg(tokens,opts);
    mdl = extract_model(tokens);

    % Append
    config_list{end+1} = cfg;
    model_list{end+1} = mdl;
    context_list{end+1} = context;
    accuracy_list(end+1) = acc;
end

% Create summary table
summaryT = table(config_list', model_list', context_list', accuracy_list', ...
    'VariableNames', {'Config', 'Model', 'Context', 'Accuracy'});

% Set categorical order
base_feats = sprintf('%d', opts.num_features);
combined_feats = sprintf('%dwCsp', opts.num_features);

cfg_order = {base_feats, 'csp', combined_feats};   
model_order = {'STANDARD', 'HYPER', 'NORM', 'HYPER NORM'};
ctx_order = {'Within', 'Cross'};

summaryT.Config = categorical(summaryT.Config, cfg_order, 'Ordinal', true);
summaryT.Model = categorical(summaryT.Model, model_order, 'Ordinal', true);
summaryT.Context = categorical(summaryT.Context, ctx_order, 'Ordinal', true);

% Compute grouped stats
grouped = groupsummary(summaryT, {'Config','Model','Context'}, ...
    {'mean', 'std', 'median'}, 'Accuracy');
grouped.Properties.VariableNames{'mean_Accuracy'} = 'MeanAccuracy';
grouped.Properties.VariableNames{'std_Accuracy'} = 'StdAccuracy';
grouped.Properties.VariableNames{'median_Accuracy'} = 'MedianAccuracy';
grouped = sortrows(grouped, {'Config', 'Model', 'Context'});

% Write grouped stats to Excel
output_file = sprintf('PreCalib_%dsamples_Stats.xlsx', opts.total_samples);
writetable(grouped, output_file, 'Sheet', 'Basics');
fprintf('[INFO] Accuracy stats written to: %s (Sheet: Basics)\n', output_file);


% -------------------------------------------------------------------------
% Boxplots: One for Within, one for Cross
% -------------------------------------------------------------------------
contexts = {'Within', 'Cross'};
for i = 1:numel(contexts)
    ctx = contexts{i};
    ctxT = summaryT(summaryT.Context == ctx, :);

    figure('Name', sprintf('Accuracy Boxplots (%s)', ctx), ...
        'NumberTitle', 'off', ...
        'Renderer', 'painters'); % optional for saving/exporting

    sgtitle(sprintf('Accuracy Boxplots (%s) | N = %d samples', ctx, opts.total_samples), 'FontWeight', 'bold');

    % Config boxplot
    subplot(1,2,1)
    boxplot(ctxT.Accuracy, ctxT.Config)
    title(sprintf('Boxplot by Config (%s)', ctx))
    ylabel('Accuracy (%)')
    grid on

    % Model boxplot
    subplot(1,2,2)
    boxplot(ctxT.Accuracy, ctxT.Model)
    title(sprintf('Boxplot by Model Type (%s)', ctx))
    ylabel('Accuracy (%)')
    grid on
end


% ----------------------------
% ANOVA + POSTHOC for Within/Cross separately
% ----------------------------
contexts = {'Within', 'Cross'};
anova_results = {};
posthoc_rows = {};

for c = 1:numel(contexts)
    ctx = contexts{c};
    subset = summaryT(summaryT.Context ==ctx, :);
    if isempty(subset), continue; end

    cfg_var = categorical(subset.Config);
    mdl_var = categorical(subset.Model);

    fprintf('\n[INFO] Running 2-way ANOVA for context: %s\n', ctx);
    [p, tbl, stats] = anovan(subset.Accuracy, ...
        {cfg_var, mdl_var}, ...
        'model', 2, ...
        'varnames', {'Config','Model'}, ...
        'display', 'off');

    % Effect sizes (η²)
    SS_total = tbl{end,2};
    eta_sq_cfg = tbl{2,2} / SS_total;
    eta_sq_mdl = tbl{3,2} / SS_total;
    eta_sq_inter = tbl{4,2} / SS_total;

    % Show result
    fprintf('[%s ANOVA] p(Config)=%.4f, η²=%.3f | p(Model)=%.4f, η²=%.3f | p(Int)=%.4f, η²=%.3f\n', ...
        ctx, p(1), eta_sq_cfg, p(2), eta_sq_mdl, p(3), eta_sq_inter);

    % Store summary
    anova_results(end+1,:) = {ctx, p(1), eta_sq_cfg, p(2), eta_sq_mdl, p(3), eta_sq_inter};

    % --------------------------
    % POSTHOC (only if ANOVA significant)
    % --------------------------
    if p(1) < 0.05
        fprintf('\n[POST-HOC] Config differences (%s):\n', ctx);

        % Show plot
        figure('Name', sprintf('Post-hoc Config (%s)', ctx));
        multcompare(stats, 'Dimension', 1, 'Display', 'on');
        title(['Post-hoc: Config (' ctx ')']);

        % Get table for saving
        comp = multcompare(stats, 'Dimension', 1, 'Display', 'off');
        labels = categories(cfg_var);
        for j = 1:size(comp,1)
            posthoc_rows(end+1,:) = {ctx, 'Config', labels{comp(j,1)}, labels{comp(j,2)}, comp(j,6), comp(j,6)<0.05};
        end
    end

    if p(2) < 0.05
        fprintf('\n[POST-HOC] Model differences (%s):\n', ctx);

        % Show plot
        figure('Name', sprintf('Post-hoc Model (%s)', ctx));
        multcompare(stats, 'Dimension', 2, 'Display', 'on');
        title(['Post-hoc: Model (' ctx ')']);

        % Get table for saving
        comp = multcompare(stats, 'Dimension', 2, 'Display', 'off');
        labels = categories(mdl_var);
        for j = 1:size(comp,1)
            posthoc_rows(end+1,:) = {ctx, 'Model', labels{comp(j,1)}, labels{comp(j,2)}, comp(j,6), comp(j,6)<0.05};
        end
    end

    % Effect Size Plot
    figure('Name', sprintf('Effect Sizes (%s)', ctx), 'NumberTitle', 'off');
    bar([eta_sq_cfg, eta_sq_mdl, eta_sq_inter]);
    set(gca, 'XTickLabel', {'Config', 'Model', 'Interaction'});
    xlabel('Factors');
    ylabel('η² (Effect Size)');
    title(['Effect Sizes for ' ctx ' Context']);
    ylim([0 1]); grid on;
end


% -------------------------------------------------------------------------
% Write ANOVA and Post-Hoc to Excel
% -------------------------------------------------------------------------
% Rename columns for clarity and consistency
anovaT = cell2table(anova_results, ...
    'VariableNames', {
        'Context', ...
        'Config_p', ...
        'Config_Eta2', ...
        'Model_p', ...
        'Model_Eta2', ...
        'Interaction_p', ...
        'Interaction_Eta2'
    });

% Tooltip row with explanations
tooltipRow = {
    'Type of analysis performed (Within or Cross)', ...
    'p-value for effect of feature config (e.g. 24 vs csp)', ...
    'η² effect size for config (variance explained by config)', ...
    'p-value for effect of model type (e.g. SVM variants)', ...
    'η² effect size for model type (variance explained by model)', ...
    'p-value for config × model interaction effect', ...
    'η² effect size of interaction (variance explained by interaction)'
};

% Write tooltip to row 1
writecell(tooltipRow, output_file, 'Sheet', 'ANOVA', 'Range', 'A1');

% Write table below it starting at row 2
writetable(anovaT, output_file, 'Sheet', 'ANOVA', 'Range', 'A2');

% Save post-hoc compact table
if isempty(posthoc_rows)
    no_posthoc_msg = {
        'No significant ANOVA effects were found for either feature configuration or model type.';
        'Therefore, post-hoc pairwise comparisons were not performed.';
    };
    writecell(no_posthoc_msg, output_file, 'Sheet', 'Posthoc', 'Range', 'A1');
else 
    % Convert to table with clean column headers
    posthocT = cell2table(posthoc_rows, ...
        'VariableNames', {
            'Context', ...   % Formerly 'Context'
            'EffectType', ...     % Formerly 'Effect'
            'GroupA', ...
            'GroupB', ...
            'p_value', ...
            'IsSignificant'
        });

    % Tooltip row for better readability in Excel
    tooltipRow = {
        'Type of analysis performed (Within or Cross)', ...
        'Effect being tested (Config or Model)', ...
        'First group in the comparison', ...
        'Second group in the comparison', ...
        'p-value for this pairwise comparison', ...
        'Significance flag (1 = p < 0.05, else 0)'
    };

    % Write tooltips and table
    writecell(tooltipRow, output_file, 'Sheet', 'Posthoc', 'Range', 'A1');
    writetable(posthocT, output_file, 'Sheet', 'Posthoc', 'Range', 'A2');
end

fprintf('[INFO] ANOVA and Post-Hoc results added to Excel file: %s\n', output_file);

end


% -------------------------------------------------------------------------
% Helper Functions
% -------------------------------------------------------------------------
function cfg = extract_cfg(tokens,opts)
base_feats = sprintf('%d', opts.num_features);
combined_feats = sprintf('%dwCsp', opts.num_features);
valid_cfgs = {base_feats, 'csp', combined_feats};
cfg = 'Unknown';
for i = 1:length(tokens)
    if any(strcmpi(tokens{i}, valid_cfgs))
        cfg = tokens{i};
        return;
    end
end
end

function mdl = extract_model(tokens)
valid_mdls = {'STANDARD', 'HYPER', 'NORM', 'HYPER NORM'};
mdl = 'Unknown';
for i = 1:length(tokens)
    rest = strjoin(tokens(i:end), ' ');
    if any(strcmp(rest, valid_mdls))
        mdl = rest;
        return;
    elseif any(strcmp(tokens{i}, valid_mdls))
        mdl = tokens{i};
        return;
    end
end


end
