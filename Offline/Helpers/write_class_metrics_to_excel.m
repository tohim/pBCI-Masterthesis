function write_class_metrics_to_excel(per_class_table, source_tag, target_tag, model_label, filename)
%WRITE_CLASS_METRICS_TO_EXCEL Writes per-class metrics to an Excel sheet
%   Each sheet is named based on source dataset and feature config.

    % Create a copy of the class metrics and add context info
    T = per_class_table;
    T.Source = repmat({source_tag}, height(T), 1);
    T.Target = repmat({target_tag}, height(T), 1);
    T.Model  = repmat({model_label}, height(T), 1);
    
    % Reorder: Move 'Class' to first, then Source/Target/Model right after
    if ismember('Class', T.Properties.VariableNames)
        T = movevars(T, 'Class', 'Before', 1);
    end
    T = movevars(T, {'Source', 'Target', 'Model'}, 'After', 'Class');
    
    if startsWith(source_tag, 'MATB_easy_diff')
        source_tag = "EasyDiff";
    elseif startsWith(source_tag, 'MATB_easy_meddiff')
        source_tag = 'EasyMedDiff';
    elseif startsWith(source_tag, 'STEW')
        source_tag = 'STEW';
    elseif startsWith(source_tag, 'HEAT')
        source_tag = 'HEATCHAIR';
    end

    % Define sheet name
    sheet_name = sprintf('Classes_%s', source_tag);
    sheet_name = matlab.lang.makeValidName(sheet_name);  % Ensure compatibility

    % Append table to Excel
    writetable(T, filename, 'Sheet', sheet_name, 'WriteMode', 'append');
    
    fprintf('[SAVED] Per-class metrics → %s (sheet: %s)\n', filename, sheet_name);
end