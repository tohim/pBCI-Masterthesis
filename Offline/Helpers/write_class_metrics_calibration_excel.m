function write_class_metrics_calibration_excel(per_class_table, source_tag, target_tag, model_label, calib_type, filename)

    % Add CalibrationType column
    per_class_table.CalibrationType = repmat({calib_type}, height(per_class_table), 1);

    % Reorder CalibrationType to appear after Model (if it exists)
    if ismember('Model', per_class_table.Properties.VariableNames)
        per_class_table = movevars(per_class_table, 'CalibrationType', 'After', 'Model');
    end

    % Ensure CalibrationType is a categorical with fixed order
    calib_order = {'adapted', 'finetuned', 'finetuned_adapted'};
    per_class_table.CalibrationType = categorical(per_class_table.CalibrationType, calib_order, 'Ordinal', true);

    % Now sort by CalibrationType (ordinal sort will respect the order!)
    per_class_table = sortrows(per_class_table, 'CalibrationType');

    % Write to Excel via wrapper
    write_class_metrics_to_excel(per_class_table, source_tag, target_tag, model_label, filename);

end
