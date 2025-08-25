% === Helper: Add entry to result table ===

function results = add_result(results, name, column, accuracy)
    % Create new row as a table
    new_row = table(string(name), string(column), round(accuracy,2), ...
                    'VariableNames', {'SOURCE', 'TARGET', 'ACCURACY'});
    
    % Append to results
    results = [results; new_row];
end