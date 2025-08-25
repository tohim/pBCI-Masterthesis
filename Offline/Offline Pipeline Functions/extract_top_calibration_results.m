function extract_top_calibration_results(filename, config_tags, threshold)
    % Collect top results across all config-specific calibration result sheets

    all_results = [];

    for i = 1:length(config_tags)
        tag = config_tags{i};
        sheetname = ['CalibResults_' tag];
        
        try
            T = readtable(filename, 'Sheet', sheetname, 'ReadRowNames', true);
        catch
            warning('[SKIP] Could not read sheet: %s', sheetname);
            continue;
        end

        [nRows, nCols] = size(T);
        rowNames = T.Properties.RowNames;
        colNames = T.Properties.VariableNames;

        for r = 1:nRows
            for c = 1:nCols
                cell_str = T{r, c}{1};
                if isempty(cell_str), continue; end

                % Flatten any multiline text
                flat_str = regexprep(cell_str, '\s+', ' ');

                % Extract the cross accuracy
                match = regexp(flat_str, 'Cross:\s*([\d.]+)%', 'tokens');
                if ~isempty(match)
                    acc = str2double(match{1}{1});
                    if acc >= threshold
                        new_entry = table;
                        new_entry.Source = string(rowNames{r});
                        new_entry.Target = string(colNames{c});
                        new_entry.Config = string(tag);
                        new_entry.CrossAccuracy = acc;
                        all_results = [all_results; new_entry];
                    end
                end
            end
        end
    end

    if isempty(all_results)
        fprintf('[INFO] No Cross Accuracy ≥ %.1f%% found across sheets.\n', threshold);
    else
        
        % Sort by CrossAccuracy descending
        all_results = sortrows(all_results, 'CrossAccuracy', 'descend');

        writetable(all_results, filename, 'Sheet', 'Top_CalibResults');
        fprintf('Top results saved to sheet: Top_CalibResults\n');
    end
end
