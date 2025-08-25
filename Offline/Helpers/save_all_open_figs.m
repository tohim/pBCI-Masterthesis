function save_all_open_figs(save_path)
% save_all_open_figs - Saves all open figures as .fig files to specified path
%
% Usage:
%   save_all_open_figs('C:\your\desired\folder')

    if ~exist(save_path, 'dir')
        mkdir(save_path);
    end

    figs = findall(0, 'Type', 'figure');
    for i = 1:length(figs)
        fig = figs(i);
        filename = fullfile(save_path, sprintf('Figure_%d.fig', fig.Number));
        savefig(fig, filename);
    end
end
