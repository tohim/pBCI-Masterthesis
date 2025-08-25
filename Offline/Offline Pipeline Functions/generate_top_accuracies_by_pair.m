function generate_top_accuracies_by_pair(file_path, sample_size, threshold)

    % Set default threshold to 70 if not specified
    if nargin < 3
        threshold = 70;
    end

    % Load the "Total" sheet
    T = readtable(file_path, 'Sheet', 'Total');

    % Filter: only keep rows with accuracy >= threshold
    T = T(T.ACCURACY >= threshold, :);

    % Group by SOURCE and TARGET, and get the row with the max accuracy
    [unique_pairs, ~, group_idx] = unique(strcat(T.SOURCE, '->', T.TARGET));
    best_rows = zeros(length(unique_pairs), 1);

    for i = 1:length(unique_pairs)
        group_rows = find(group_idx == i);
        [~, max_idx] = max(T.ACCURACY(group_rows));
        best_rows(i) = group_rows(max_idx);
    end

    % Extract the top rows
    top_table = T(best_rows, :);

    % Sort by accuracy descending
    top_table = sortrows(top_table, 'ACCURACY', 'descend');

    % Define sheet name using sample size
    sheet_name = sprintf('TopAcc_ByPair_%d', sample_size);

    % Write to the same Excel file
    writetable(top_table, file_path, 'Sheet', sheet_name);

    fprintf('Top Accuracy Summary written to sheet: %s (min %.0f%% accuracy)\n', ...
        sheet_name, threshold);
end