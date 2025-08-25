function f_scores = compute_anova_f(X, y)
%COMPUTE_ANOVA_F Computes F-scores for each feature based on ANOVA
%   Assumes binary labels y (0 and 1)

    n_features = size(X, 2);
    f_scores = zeros(1, n_features);

    for i = 1:n_features
        x = X(:, i);
        group0 = x(y == 0);
        group1 = x(y == 1);

        % Between-group variance
        mean_total = mean(x);
        n0 = numel(group0);
        n1 = numel(group1);
        mean0 = mean(group0);
        mean1 = mean(group1);
        ss_between = n0*(mean0 - mean_total)^2 + n1*(mean1 - mean_total)^2;

        % Within-group variance
        ss_within = sum((group0 - mean0).^2) + sum((group1 - mean1).^2);

        f_scores(i) = ss_between / (ss_within + eps); % add eps to avoid division by zero
    end
end