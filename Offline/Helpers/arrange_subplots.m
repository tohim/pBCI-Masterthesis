%% Re arrange subplots

%% Boxplots

% === Work on the CURRENT figure (no re-plot) ===
fig = gcf;


% Find both axes and order them left→right by their position
axs = findall(fig,'Type','axes');                 % should be 2 axes here

set(axs(2), 'Position', [0.08 0.15 0.4 0.75]); % left boxplot
set(axs(1), 'Position', [0.55 0.15 0.4 0.75]); % right boxplot

[~, idx] = sort(arrayfun(@(a)a.Position(1), axs));
axs = axs(idx);
ax  = axs(1);   % Config (3 categories)
axRight = axs(2);   % Model  (4 categories)

% Clear the supertitle text (works whether or not one already exists)
sgtitle('');
h = sgtitle('');
h.Visible = 'off';

% ----- Rename category labels on the x-axes -----
% Left panel (feature/config)
ax.XTick = 1:3;
ax.XTickLabel = {'Base-only','CSP-only','Base+CSP'};
ax.XTickLabelRotation = 0;   % optional

% Right panel (model type)
axRight.XTick = 1:4;
axRight.XTickLabel = {'Standard','Hyper','Norm','HyperNorm'};
axRight.XTickLabelRotation = 0;  % optional

% ----- Make labels/ticks/titles bigger -----
set([ax, axRight], 'FontSize',18, 'LineWidth',1.1); % ticks & category labels
ax.XLabel.FontSize  = 25;   ax.YLabel.FontSize  = 18;   ax.Title.FontSize  = 15;
axRight.XLabel.FontSize = 25;   axRight.YLabel.FontSize = 18;   axRight.Title.FontSize = 15;

% If you used sgtitle, enlarge it too
h = findall(fig,'Type','Text','Tag','suptitle');
if ~isempty(h), set(h,'FontSize',20,'FontWeight','bold'); end



% (Optional) enlarge canvas and export
set(fig,'Units','centimeters','Position',[2 2 30 16]);
% exportgraphics(fig,'accuracy_boxplots.png','Resolution',300);
% exportgraphics(fig,'accuracy_boxplots.pdf');   % vector



%% Post-hocs


fig = gcf;

% Pick the main axes (largest width), robust if other UI axes exist
axList = findall(fig,'Type','axes');
[~,idx] = max(arrayfun(@(a)a.Position(3), axList));
ax = axList(idx);

% % -------- Rename Y tick labels --------
% oldLabs = string(ax.YTickLabel);                 % e.g., "Config=25", "Config=csp", ...
% keys    = ["25","csp","25wCsp"];
% vals    = ["Base-only","CSP-only","Base+CSP"];
% 
% % Strip "Config=", map each key to its pretty label, keep current order
% core = erase(oldLabs, "Config=");
% newLabs = strings(size(core));
% for i = 1:numel(core)
%     k = find(keys == core(i), 1);
%     if ~isempty(k), newLabs(i) = vals(k); else, newLabs(i) = oldLabs(i); end
% end
% ax.YTickLabel = newLabs;

% -------- Rename Y tick labels --------
oldLabs = string(ax.YTickLabel);                 % e.g., "Config=Standard", "Config=Hyper", ...
keys    = ["STANDARD","HYPER","NORM","HYPER NORM"];
vals    = ["STANDARD","HYPER","NORM","HYPERNORM"];

% Strip "Model=", map each key to its pretty label, keep current order
core = erase(oldLabs, "Model=");
newLabs = strings(size(core));
for i = 1:numel(core)
    k = find(keys == core(i), 1);
    if ~isempty(k), newLabs(i) = vals(k); else, newLabs(i) = oldLabs(i); end
end
ax.YTickLabel = newLabs;

% Optional: informative y-axis label
%ylabel(ax, 'Config (Base-only, CSP-only, Base+CSP)');

% -------- Make it more readable --------
ax.FontSize   = 18;       % tick labels & Y tick text
ax.LineWidth  = 1.2;
ax.Title.FontSize = 16;     % remove subplot title if you don't want it
% If there's an sgtitle from earlier and you want it gone:
% hS = findall(fig,'Type','Text','Tag','suptitle');
% if ~isempty(hS), delete(hS); end

% Thicken the confidence-interval lines and markers
ln = findall(ax,'Type','Line');       % the CI bars + markers are 'line' objects
set(ln, 'LineWidth', 3, 'MarkerSize', 10);  % bolder bars and larger circles

% Give the plot more space inside the figure
ax.Position = [0.12 0.16 0.83 0.76];  % [left bottom width height], tweak as you like

% (Optional) make the canvas larger and export
set(fig,'Units','centimeters','Position',[2 2 30 16]);


fig = gcf;

% 1) Fix Y-tick labels (you already had this)
axList = findall(fig,'Type','axes');
[~,idx] = max(arrayfun(@(a)a.Position(3), axList));   % main axes
ax = axList(idx);

oldLabs = string(ax.YTickLabel);
keys = ["25","csp","25wCsp"];
vals = ["Base-only","CSP-only","Base+CSP"];
core = erase(oldLabs,"Config=");
newLabs = oldLabs;
for i = 1:numel(core)
    k = find(keys==core(i),1);
    if ~isempty(k), newLabs(i) = vals(k); end
end
ax.YTickLabel = newLabs;

% 2) Fix the subtitle text at the bottom
txt = findall(fig,'Type','Text');            % all text objects in the figure
for h = txt'
    s = string(get(h,'String'));             % might be char/cell -> normalize
    % Replace any "Config=..." occurrences
    s = regexprep(s, "Config=25wCsp", "Base+CSP");
    s = regexprep(s, "Config=csp",   "CSP-only");
    s = regexprep(s, "Config=25",    "Base-only");
    % If you ever have Model=...
    s = regexprep(s, "Model=Standard", "STANDARD");
    s = regexprep(s, "Model=HYPER", "HYPER");
    s = regexprep(s, "Model=NORM", "NORM");
    s = regexprep(s, "Model=HYPER NORM", "HYPERNORM");
    set(h,'String',s);
end

% 3) (Optional) styling
set(ax,'FontSize',18,'LineWidth',1.2);
set(findall(fig,'Type','Line'),'LineWidth',3,'MarkerSize',10);
set(fig,'Units','centimeters','Position',[2 2 30 16]);




%% MEAN plots

% === Apply to the CURRENT figure (no re-plot) ===
fig = gcf;

% Grab the main axes (ignore legend axes)
axList = findall(fig,'Type','axes');
axList = axList(~strcmp(get(axList,'Tag'),'legend'));
[~,idx] = max(arrayfun(@(a)a.Position(3), axList));
ax = axList(idx);

% ---------- Typography ----------
ax.FontSize = 13;                 % ticks & category labels
ax.LineWidth = 1.2;
ax.XLabel.FontSize = 14;
ax.Title.FontSize  = 15;
%ax.YAxis.FontWeight = 'bold';     % thicker Y tick labels
ax.TickLabelInterpreter = 'none'; % keep underscores literal

% ---------- Thicken bars ----------
b = findall(ax,'Type','Bar');     % both series from barh(...)
set(b, 'BarWidth', 0.9, 'EdgeColor','none');  % thicker, no outlines

% ---------- Use all available space ----------
% Compute tight position that keeps room for tick labels and title
ti = ax.TightInset;               % [left bottom right top] padding needed
pad = 0.05;                       % tiny extra padding
ax.Position = [ti(1)+pad, ti(2)+pad, ...
               1 - ti(1) - ti(3) - 2*pad, ...
               1 - ti(2) - ti(4) - 2*pad];

% ---------- Reduce vertical whitespace ----------
N = numel(ax.YTick);
ylim(ax, [0.5, N+0.5]);           % trim top/bottom margins

% Optional smaller grid aesthetics
grid(ax,'on'); ax.YGrid = 'off';  % only x-grid
ax.GridLineStyle = ':'; ax.GridAlpha = 0.3; box(ax,'off');

% ---------- Legend tweaks (if present) ----------
lgd = findobj(fig,'Type','Legend');
if ~isempty(lgd)
    set(lgd, 'FontSize', 12, 'Box','off', 'Location','southoutside');
end

% ---------- Nice figure size for theses + export ----------
set(fig,'Units','centimeters','Position',[2 2 16 15]);  % W×H; adjust to taste


%% NASA TLX Boxplot



% === Make CURRENT boxplot figure thesis-ready ===
fig = gcf;

% Canvas size
set(fig, 'Units','centimeters', 'Position',[2 2 30 16]);

% Main axes (largest width if multiple exist)
axList = findall(fig,'Type','axes');
[~, idx] = max(arrayfun(@(a)a.Position(3), axList));
ax = axList(idx);

% Axes fonts/lines
ax.FontSize   = 18;
ax.LineWidth  = 1.2;
ax.XLabel.FontSize = 20;
ax.YLabel.FontSize = 20;
ax.Title.FontSize  = 18;

% Fill the figure (respecting tight insets)
in = ax.TightInset;                         % [L B R T]
ax.Position = [0.08+in(1), 0.10+in(2), ...
               0.90-in(1)-in(3), 0.86-in(2)-in(4)];

% Thicken boxplot glyphs
set(findobj(ax,'Tag','Box'),     'LineWidth',1.8);
set(findobj(ax,'Tag','Median'),  'LineWidth',2.2);
set(findobj(ax,'Tag','Whisker'), 'LineWidth',1.5);
set(findobj(ax,'Tag','Outliers'),'MarkerSize',6);

% (Optional) rotate crowded category labels
% ax.XTickLabelRotation = 20;

% (Optional) legend font
set(findall(fig,'Type','Legend'), 'FontSize',16);

% (Optional) export
% exportgraphics(fig,'boxplot_thesis.png','Resolution',300);
% exportgraphics(fig,'boxplot_thesis.pdf');








%% NASA Hist

% === Make CURRENT histogram figure thesis-ready ===
fig = gcf;

% Canvas size
set(fig, 'Units','centimeters', 'Position',[2 2 30 16]);

% Axes styling (handles single/multiple axes)
axs = findall(fig, 'Type','axes');
for k = 1:numel(axs)
    ax = axs(k);
    ax.FontSize   = 18;      % tick & category labels
    ax.LineWidth  = 1.2;     % axes border
    ax.XLabel.FontSize = 20;
    ax.YLabel.FontSize = 20;
    ax.Title.FontSize  = 18;

    % Fill the figure (respecting tight insets)
    in = ax.TightInset;                         % [L B R T]
    ax.Position = [0.08+in(1), 0.10+in(2), ...
                   0.90-in(1)-in(3), 0.86-in(2)-in(4)];
end

% Thicken bars (works for histogram() and bar())
set(findall(fig,'Type','Histogram'), 'LineWidth',1.5);
set(findall(fig,'Type','Bar'),       'LineWidth',1.5);

% (Optional) legend font
set(findall(fig,'Type','Legend'), 'FontSize',16);

% (Optional) export
% exportgraphics(fig,'histogram_thesis.png','Resolution',300);
% exportgraphics(fig,'histogram_thesis.pdf');












