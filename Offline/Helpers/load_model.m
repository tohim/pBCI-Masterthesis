function mdl = load_model(fname)
    if exist(fname, 'file')
        data = load(fname);
        varnames = fieldnames(data);
        mdl_idx = find(contains(varnames, 'mdl'), 1);
        if isempty(mdl_idx)
            error('No model variable found in file: %s', fname);
        end
        mdl = data.(varnames{mdl_idx});
    else
        error('Model file not found: %s', fname);
    end
end
