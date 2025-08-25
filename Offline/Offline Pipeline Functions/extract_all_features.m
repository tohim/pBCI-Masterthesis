function [train_features, val_features, test_features, W_csp] = extract_all_features(train_labels, val_labels, test_labels, train_epochs, val_epochs, test_epochs, opts)

    % Initializations
    feature_tag = '';
    train_features = []; val_features = []; test_features = [];
    W_csp = [];

    % Epoch Length String
    epoch_length = sprintf('%dsec', opts.epochlength);

    % Handle sample count prefix for saving
    if isfield(opts, 'total_samples') && opts.total_samples > 0
        sample_tag = sprintf('%d', opts.total_samples);
    else
        sample_tag = 'full';
    end

% -------------------------------------------------------------------------
% HANDCRAFTED FEATURE EXTRACTION
% -------------------------------------------------------------------------
    if opts.use_features
        fprintf('\n[INFO] Using handcrafted features...\n');
        feature_str = sprintf('%d', opts.num_features); % e.g., '24', '16', etc.
        tag_feat = [sample_tag '_' feature_str '_' epoch_length '_' opts.proc '_' opts.dataset];

        if exist([tag_feat '_train_features.mat'], 'file')
            train = load([tag_feat '_train_features.mat']); 
            train = train.train_features;
            val   = load([tag_feat '_val_features.mat']);   
            val   = val.val_features;
            test  = load([tag_feat '_test_features.mat']);  
            test  = test.test_features;
            fprintf('[INFO] Loaded handcrafted features from file.\n');
        else
            fprintf('[INFO] Extracting handcrafted features...\n');
            train = OFF_extract_features(train_epochs, opts);
            val   = OFF_extract_features(val_epochs, opts);
            test  = OFF_extract_features(test_epochs, opts);
            save([tag_feat '_train_features.mat'], 'train_features');
            save([tag_feat '_val_features.mat'],   'val_features');
            save([tag_feat '_test_features.mat'],  'test_features');
        end
        feature_tag = feature_str;
    end

% -------------------------------------------------------------------------
% CSP FEATURE SECTION | Csp Filter Creation
% -------------------------------------------------------------------------
    if opts.use_csp
        fprintf('\n[INFO] Using CSP features...\n');
         tag_csp = sprintf('%s_csp_%d_%dsec_%s_%s', sample_tag, opts.num_csp_filters, opts.epochlength, opts.proc, opts.dataset);

         if exist([tag_csp '_train_features.mat'], 'file')
             
             if exist([tag_csp '_W_csp.mat'], 'file')
                 temp = load([tag_csp '_W_csp.mat']);
                 W_csp = temp.W_csp;
             else
                 warning('W_csp not found in file! CSP extraction may fail later.');
                 W_csp = [];
             end

             csp_train = load([tag_csp '_train_features.mat']);
             csp_train = csp_train.train_features;
             csp_val   = load([tag_csp '_val_features.mat']);
             csp_val   = csp_val.val_features;
             csp_test  = load([tag_csp '_test_features.mat']);  
             csp_test  = csp_test.test_features;
             fprintf('[INFO] Loaded CSP features from file.\n');
            
        else
            fprintf('[INFO] Extracting CSP features...\n');
            [W_csp, ~] = train_csp(train_epochs, train_labels, opts.num_csp_filters);
            save([tag_csp '_W_csp.mat'], 'W_csp');
            csp_train = extract_csp_features(train_epochs, W_csp);
            csp_val   = extract_csp_features(val_epochs, W_csp);
            csp_test  = extract_csp_features(test_epochs, W_csp);
            save([tag_csp '_train_features.mat'], 'csp_train');
            save([tag_csp '_val_features.mat'],   'csp_val');
            save([tag_csp '_test_features.mat'],  'csp_test');
        end

        % Combine or assign
        if isempty(feature_tag)
            feature_tag = sprintf('csp_%d', opts.num_csp_filters);
            train_features = csp_train;
            val_features   = csp_val;
            test_features  = csp_test;
        else
            fprintf('\n[INFO] Combining handcrafted and CSP features...\n');
            feature_tag = sprintf('%swCsp', feature_tag);
            train_features = [train, csp_train];
            val_features   = [val,   csp_val];
            test_features  = [test,  csp_test];
        end
    else
        if opts.use_features
            train_features = train;
            val_features   = val;
            test_features  = test;
        end
    end

% -------------------------------------------------------------------------
% Save Final Combined (or Single) Feature Set
% -------------------------------------------------------------------------
    base = sprintf('%s_%s_%dsec_%s', sample_tag, feature_tag, opts.epochlength, opts.proc);
    save([base '_' opts.dataset '_train_features.mat'], 'train_features');
    save([base '_' opts.dataset '_val_features.mat'],   'val_features');
    save([base '_' opts.dataset '_test_features.mat'],  'test_features');

    
    % Final Checks
    if size(train_features,1) ~= size(train_labels)
        error('Mismatch between number of train epochs and train labels.');
    end
    if size(val_features,1) ~= size(val_labels)
        error('Mismatch between number of val epochs and val labels.');
    end
    if size(test_features,1) ~= size(test_labels)
        error('Mismatch between number of test epochs and test labels.');
    end

    if opts.use_csp && isempty(W_csp)
        error('CSP is enabled, but W_csp is empty. Make sure CSP filters were computed.');
    end

end
