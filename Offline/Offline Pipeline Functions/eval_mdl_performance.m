function [accuracy, per_class_table] = eval_mdl_performance(model, features, labels, class_labels, context_str, verbose)

% -------------------------------------------------------------------------
% Function to Evaluate the Model Performance on Val or Test Data
% labels = y_true      :    Ground Truth Labels (Nx1)
% predictions = y_pred :    Predicted Labels (Nx1)
% class_labels         :    Cell array of class names (e.g., {'Low','High'})
% context_str          :    Name of the current Evaluation Block
% verbose              :    Toggle prints and figure output (true/false)
% -------------------------------------------------------------------------

% Default verbose ON
if nargin < 6
    verbose = true;
end

% Predictions
predictions = predict(model, features);

if nargin < 4 || isempty(class_labels)
    unique_classes = unique([labels;predictions]);
    if isequal(unique_classes, [0;1]) || isequal(unique_classes, [1;0])
        class_labels = {'Low', 'High'};
    else
        class_labels = cellstr(string(unique_classes));
    end
end

if nargin < 5
    context_str = 'Model Evaluation';
end

% Check size mismatch
if length(labels) ~= length(predictions)
    error('Labels and Predictions must be same length.');
end

% Confusion Matrix
[C, order] = confusionmat(labels, predictions);
TP = C(2,2); FP = C(1,2); FN = C(2,1); TN = C(1,1);

% Compute Accuracy
accuracy = (TP + TN) / sum(C(:));

% Only display if verbose
if verbose
    box_width = 60;
    fprintf('\n%s\n', repmat('=', 1, box_width));
    fprintf(' %-46s \n', ['Evaluation: ', context_str]);
    fprintf('%s\n', repmat('=', 1, box_width));

    fprintf('\n%-50s\n', '------------- Total Model Performance ------------');
    fprintf('     %-21s : %.2f\n', 'Accuracy', accuracy * 100);
end

% Compute Summary Metrics
precision = round((TP / (TP + FP)), 2);
recall    = round((TP / (TP + FN)), 2);
specificity = round((TN / (TN + FP)), 2);
f1       = round((2 * (precision * recall) / (precision + recall)), 2);

if verbose
    fprintf('     %-21s : %.2f\n', 'Precision (Quality)', precision);
    fprintf('     %-21s : %.2f\n', 'Recall (Sensitivity)', recall);
    fprintf('     %-21s : %.2f\n', 'Specificity', specificity);
    fprintf('     %-21s : %.2f\n','F1 Score', f1);

    fprintf('\n%-50s\n', '---------- Individual Class Performance ----------');
    fprintf('\nClass Distribution (Ground Truth):\n');
end

% Class Metrics
num_classes = length(order);
precision = zeros(num_classes,1);
recall    = zeros(num_classes,1);
f1        = zeros(num_classes,1);

for i = 1:num_classes
    TP = C(i,i);
    FP = sum(C(:,i)) - TP;
    FN = sum(C(i,:)) - TP;
    precision(i) = round((TP / (TP + FP + eps)), 2);
    recall(i)    = round((TP / (TP + FN + eps)), 2);
    f1(i)        = round((2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps)), 2);
end

% Print per-class if verbose
if verbose
    for i = 1:num_classes
        class_count = sum(labels == order(i));
        fprintf('     %-21s : %d samples (%.1f%%)\n', class_labels{i}, class_count, 100 * class_count / length(labels));
        fprintf('   Class: %s\n', class_labels{i});
        fprintf('     %-21s : %.2f\n', 'Precision', precision(i));
        fprintf('     %-21s : %.2f\n', 'Recall (Sensitivity)', recall(i));
        fprintf('     %-21s : %.2f\n', 'F1 Score', f1(i));
    end

    fprintf('\n%s\n%s\n\n', repmat('=', 1, box_width), repmat('=', 1, box_width));
end

% Output Table
per_class_table = table(class_labels(:), precision, recall, f1, ...
    'VariableNames', {'Class', 'Precision', 'Recall', 'F1'});

% Plot Confusion Matrix
if verbose
    figure;
    conf_chart = confusionchart(C, class_labels);
    conf_chart.Title = ['Confusion Matrix - ', context_str];
    conf_chart.RowSummary = 'row-normalized';
    conf_chart.ColumnSummary = 'column-normalized';
    conf_chart.FontSize = 20;
    conf_chart.FontName = 'Arial';
    conf_chart.GridVisible = 'on';
end

end
