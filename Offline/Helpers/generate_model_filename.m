function fname = generate_model_filename(opts, hyper, norm)

    % Feature tag                                                              
    if opts.use_features && opts.use_csp
        tag = sprintf('%dwCsp_%dsec_%s', opts.num_features, opts.epochlength, opts.proc);
    elseif opts.use_features
        tag = sprintf('%d_%dsec_%s', opts.num_features, opts.epochlength, opts.proc);
    elseif opts.use_csp
        tag = sprintf('csp_%d_%dsec_%s', opts.num_csp_filters, opts.epochlength, opts.proc);
    else
        error('Invalid feature combination. Must use at least handcrafted or CSP features.');
    end

    % Sample prefix
    if isfield(opts, 'total_samples') && opts.total_samples > 0
        prefix = sprintf('%d', opts.total_samples);
    else
        prefix = 'full';
    end

    % Model type suffix (no "hyper" here!)
    if norm
        suffix = 'norm_model.mat';
    else
        suffix = 'model.mat';
    end

    % Final filename: "hyper_" prefix if needed
    if hyper
        fname = ['hyper_', prefix, '_', tag, '_', opts.dataset, '_', suffix];
    else
        fname = [prefix, '_', tag, '_', opts.dataset, '_', suffix];
    end
end
