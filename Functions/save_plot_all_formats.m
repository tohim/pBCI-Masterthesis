function save_plot_all_formats(fig_handle, base_filename)
    % Save MATLAB figure in FIG, PDF, and EPS formats
    savefig(fig_handle, base_filename + ".fig");
    exportgraphics(fig_handle, base_filename + ".pdf", 'ContentType', 'vector');
    print(fig_handle, base_filename + ".eps", '-depsc', '-vector');
end
