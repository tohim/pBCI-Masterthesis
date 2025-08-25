function features = extract_csp_features(epochs, W_csp)
% EXTRACT_CSP_FEATURES - Applies CSP filters and extracts log-variance features
% Inputs:
%   epochs   - [Channels x Time x Trials]
%   W_csp    - CSP spatial filters [Channels x num_filters]
% Outputs:
%   features - [Trials x num_filters]

    num_trials = size(epochs, 3);
    num_filters = size(W_csp, 2);
    features = zeros(num_trials, num_filters);

    for i = 1:num_trials
        X = squeeze(epochs(:,:,i));                 % [Channels x Time]
        Z = W_csp' * X;                             % Apply spatial filter → [num_filters x Time]
        var_Z = var(Z, 0, 2);                       % Variance per filter
        features(i, :) = log(var_Z / sum(var_Z));   % Log-normalized variance

        fprintf(['Saved Feature Set for Epoch Number ', num2str(i), ' / ', num2str(num_trials), '\n'])
    end

    fprintf('\nAll CSP Feature Sets for all Epochs have been saved in the corresponding Epochs x Features Matrix. \n')    
end