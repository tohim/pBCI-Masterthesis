function [acc, acc_hyper, acc_norm, acc_hyper_norm, pct_std, pct_hyper, pct_norm, pct_hyper_norm] = run_within_dataset(opts)
    
    % Load Data
    [eeg_data, labels] = load_processed_data(opts);
    [train_epochs, val_epochs, test_epochs, train_labels, val_labels, test_labels] = split_data(eeg_data, labels, opts);
    save([opts.dataset, '_QUICK_train_epochs.mat'], 'train_epochs');
    save([opts.dataset, '_QUICK_val_epochs.mat'], 'val_epochs');
    save([opts.dataset, '_QUICK_test_epochs.mat'], 'test_epochs');
    save([opts.dataset, '_QUICK_train_labels.mat'], 'train_labels');
    save([opts.dataset, '_QUICK_val_labels.mat'], 'val_labels');
    save([opts.dataset, '_QUICK_test_labels.mat'], 'test_labels');


    % Extract Features
    [train_features, val_features, test_features, W_csp] = extract_all_features(train_labels, val_labels, test_labels, train_epochs, val_epochs, test_epochs, opts);

    % Train and Evaluate Model on Validation and Test Data
    [acc, acc_hyper, acc_norm, acc_hyper_norm, pct_std, pct_hyper, pct_norm, pct_hyper_norm] = train_and_eval_models(train_features, val_features, test_features, train_labels, val_labels, test_labels, opts, W_csp);
    round(acc);
    round(acc_hyper);
    round(acc_norm);
    round(acc_hyper_norm);
end

