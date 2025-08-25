function csp_features = extract_csp_features_single_epoch(epoch_data, W_csp)
% EXTRACT_CSP_FEATURES_SINGLE_EPOCH - Applies CSP filters to a single epoch
% and extracts log-variance features
%
% Inputs:
%   epoch_data : [Channels x Time] EEG data for one epoch
%   W_csp      : [Channels x num_filters] CSP filter matrix
%
% Output:
%   csp_features : [1 x num_filters] Log-normalized variance features

    % Apply spatial filters
    Z = W_csp' * epoch_data;            % [num_filters x Time]

    % Compute variance of each CSP component
    var_Z = var(Z, 0, 2);               % [num_filters x 1]

    % Log-normalized variance
    csp_features = log(var_Z / sum(var_Z));  % [num_filters x 1]

    % Convert to row vector for consistency
    csp_features = csp_features';       % [1 x num_filters]
end
