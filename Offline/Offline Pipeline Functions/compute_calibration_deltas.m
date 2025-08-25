function T_delta = compute_calibration_deltas(T_pre, T_post)

results = [];
clean_target = @(x) strrep(x, '_finetuned', '');

for i = 1:height(T_post)
    post_row = T_post(i, :);

    % Clean target for comparison
    post_target_cleaned = clean_target(post_row.TARGET);

    % Apply cleaned matching logic
    mask = strcmp(T_pre.SOURCE, post_row.SOURCE) & ...
           strcmp(clean_target(T_pre.TARGET), post_target_cleaned) & ...
           strcmp(T_pre.Model, post_row.Model) & ...
           strcmp(T_pre.CalibrationType, post_row.CalibrationType) & ...
           strcmp(T_pre.Config, post_row.Config);

    match_idx = find(mask, 1);

    if isempty(match_idx)
        warning('[!] No PRE match found for: %s | %s | %s | %s | %s', ...
            string(post_row.SOURCE), string(post_row.TARGET), ...
            string(post_row.Model), string(post_row.CalibrationType), string(post_row.Config));
        continue;
    end

    pre_row = T_pre(match_idx, :);
    delta = post_row.ACCURACY - pre_row.ACCURACY;

    results = [results; {
        post_row.SOURCE, post_row.TARGET, post_row.Model, ...
        post_row.CalibrationType, post_row.Config, ...
        pre_row.ACCURACY, post_row.ACCURACY, delta
    }];
end

T_delta = cell2table(results, ...
    'VariableNames', {'SOURCE','TARGET','Model','CalibrationType','Config', ...
                      'PreAccuracy','PostAccuracy','Delta'});

% --- Stats ---
fprintf('[STATS] Wilcoxon Signed-Rank Test (Δ Accuracy > 0):\n');
if ~isempty(T_delta)
    [p, ~, stats] = signrank(T_delta.Delta, 0, 'tail', 'right');
    fprintf('p = %.4f | signed-rank = %d | n = %d\n', p, stats.signedrank, height(T_delta));
    fprintf('\n');
else
    fprintf('No valid entries found.\n');
end

% Sort by Delta descending (best improvements first)
T_delta = sortrows(T_delta, 'Delta', 'descend');

% Save
writetable(T_delta, 'Calibration_Delta_Results.xlsx', 'Sheet', 'Delta');
fprintf('Saved delta results to Calibration_Delta_Results.xlsx\n');

% Save Wilcoxon Stats to Sheet
if exist('p', 'var') && ~isempty(p)
    stats_table = table(p, stats.signedrank, height(T_delta), ...
        'VariableNames', {'p_value', 'signed_rank', 'N'});
else
    stats_table = table(NaN, NaN, 0, ...
        'VariableNames', {'p_value', 'signed_rank', 'N'});
end

writetable(stats_table, 'Calibration_Delta_Results.xlsx', ...
           'Sheet', 'Wilcoxon_DeltaStats', 'WriteRowNames', false);
fprintf('Saved Wilcoxon test stats to sheet: Wilcoxon_DeltaStats\n');


% -------------------------------------------------------------------------
% Summary per Calibration Type (Mean, Median, Std, Count)
% -------------------------------------------------------------------------
if ~isempty(T_delta)
    summary_stats = varfun(@mean, T_delta, ...
        'InputVariables', 'Delta', ...
        'GroupingVariables', 'CalibrationType');

    summary_stats.median = varfun(@median, T_delta, ...
        'InputVariables', 'Delta', ...
        'GroupingVariables', 'CalibrationType').median_Delta;

    summary_stats.std = varfun(@std, T_delta, ...
        'InputVariables', 'Delta', ...
        'GroupingVariables', 'CalibrationType').std_Delta;

    summary_stats.count = varfun(@numel, T_delta, ...
        'InputVariables', 'Delta', ...
        'GroupingVariables', 'CalibrationType').numel_Delta;

    % Rename columns
    summary_stats.Properties.VariableNames{'mean_Delta'} = 'Mean';
    summary_stats.Properties.VariableNames{'median'} = 'Median';
    summary_stats.Properties.VariableNames{'std'} = 'StdDev';
    summary_stats.Properties.VariableNames{'count'} = 'Count';

    % Save summary sheet
    writetable(summary_stats, 'Calibration_Delta_Results.xlsx', ...
               'Sheet', 'Delta_Summary', 'WriteRowNames', false);
    fprintf('Saved delta summary by calibration type to sheet: Delta_Summary\n');
end

end



