function [W_csp, lambda] = train_csp(epochs, labels, num_filters)
% TRAIN_CSP - Computes CSP spatial filters
% Inputs:
%   epochs       - [Channels x Time x Trials]
%   labels       - Vector of binary labels (0/1)
%   num_filters  - Total number of CSP filters (even number)
% Outputs:
%   W_csp        - CSP spatial filters [Channels x num_filters]
%   lambda       - Eigenvalues (optional)

    % Ensure binary classes
    class0_data = epochs(:,:,labels == 0);
    class1_data = epochs(:,:,labels == 1);

    % Reshape into [Channels x TotalTime]
    X1 = reshape(class0_data, size(epochs,1), []);
    X2 = reshape(class1_data, size(epochs,1), []);

    % Use built-in CSP function (from HYDRA toolbox or similar)
    [W, lambda] = csp(X1, X2);

    % Select first/last components based on num_filters
    k = num_filters / 2;
    W_csp = [W(:, 1:k), W(:, end-k+1:end)];

end