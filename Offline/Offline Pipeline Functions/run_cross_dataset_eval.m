function [acc_cross, per_class_table] = run_cross_dataset_eval(opts, mdl)
    
    % Get the Cross-Data Features and Labels
    [cross_features, cross_labels] = get_data(opts.cross_dataset, opts.cross_proc, opts);

    % Run the Model Evaluation (Testing the trained Model on Cross-Data)
    [acc_cross, per_class_table] = eval_mdl_performance(mdl, cross_features, cross_labels, [],...
        sprintf('Cross Dataset: %s → %s', opts.dataset, opts.cross_dataset), opts.verbose);
    acc_cross = round(acc_cross*100,2);
end

