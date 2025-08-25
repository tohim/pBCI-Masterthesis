%% ================== QC: whole-trace RAW vs PROCESSED (per subject) ==================
clear; close all;

n_subjects = 11;    % how many to inspect
ch = 3;             % channel index to plot (F3)
fs_raw  = 250;      % sampling rate of raw signals
fs_proc = 128;      % sampling rate of processed signals
target_pts = 2e5;   % max points we draw per trace (keeps plots responsive)

% thresholds for quick flags
flat_std_thr = 0.5;   % "flatline" if std < this
huge_std_thr = 130;   % suspiciously large std (µV)
clip_thr     = 300;   % hard clipping threshold
artifact_thr = 150;   % physiological artifact threshold

% ---- tables (now include AnyHugeProc) ----
Summary_ch = table('Size',[n_subjects 23], ...
    'VariableTypes', repmat("double",1,23), ...
    'VariableNames', {'S', ...
        'CalRawMean','CalRawStd','CalProcMean','CalProcStd', ...
        'CalRawClipPct','CalProcClipPct','CalRawArtPct','CalProcArtPct', ...
        'ExpRawMean','ExpRawStd','ExpProcMean','ExpProcStd', ...
        'ExpRawClipPct','ExpProcClipPct','ExpRawArtPct','ExpProcArtPct', ...
        'CalFlatRaw','CalFlatProc','ExpFlatRaw','ExpFlatProc', ...
        'AnyHugeRaw','AnyHugeProc'});

Summary_all = table('Size',[n_subjects 23], ...
    'VariableTypes', repmat("double",1,23), ...
    'VariableNames', Summary_ch.Properties.VariableNames);

% helper for quick stats
stats = @(x) struct( ...
    'mean',     mean(x), ...
    'std',      std(x), ...
    'clipPct',  100*mean(abs(x) > clip_thr), ...
    'artPct',   100*mean(abs(x) > artifact_thr), ...
    'isFlat',   std(x) < flat_std_thr, ...
    'isHuge',   std(x) > huge_std_thr);

% ===== main loop =====
for i = 1:n_subjects
    %% --------- Load this subject ---------
    sub_calib = load(sprintf('Subject%d_CalibrationLog.mat', i));
    sub_exp   = load(sprintf('Subject%d_Results.mat',       i));

    % 600 calib epochs (20 blocks × 30), 300 experiment epochs (10 × 30)
    cal = sub_calib.calibration_log(end-600+1:end);
    ex  = sub_exp.experiment_log(1:300);

    %% --------- Concatenate epochs: channels x time ---------
    raw_calib  = []; proc_calib = [];
    for k = 1:numel(cal)
        raw_calib  = [raw_calib,  double(cal(k).full_raw)];   % fs_raw samples/epoch
        proc_calib = [proc_calib, double(cal(k).processed)];  % fs_proc samples/epoch
    end
    raw_exp  = []; proc_exp = [];
    for k = 1:numel(ex)
        raw_exp  = [raw_exp,  double(ex(k).full_raw)];
        proc_exp = [proc_exp, double(ex(k).processed)];
    end

    % --------- KEEP NATIVE LENGTHS (no truncation) ----------
    Ncal_raw  = size(raw_calib,  2);
    Ncal_proc = size(proc_calib, 2);
    Nexp_raw  = size(raw_exp,    2);
    Nexp_proc = size(proc_exp,   2);

    % --------- Time scaling for 50% overlap ---------
    epoch_len_sec = 4;       % window length (s)
    hop_sec       = 2;       % 50% overlap -> 2 s hop
    hop_scale     = hop_sec / epoch_len_sec;   % = 0.5

    % --------- Decimation indices (per stream) ---------
    target_idx_cal_raw  = 1:max(1, floor(Ncal_raw  / target_pts)) : Ncal_raw;
    target_idx_cal_proc = 1:max(1, floor(Ncal_proc / target_pts)) : Ncal_proc;
    target_idx_exp_raw  = 1:max(1, floor(Nexp_raw  / target_pts)) : Nexp_raw;
    target_idx_exp_proc = 1:max(1, floor(Nexp_proc / target_pts)) : Nexp_proc;

    % --------- Time vectors in TRUE seconds of recording ---------
    t_cal_raw  = (target_idx_cal_raw  - 1) / fs_raw  * hop_scale;
    t_cal_proc = (target_idx_cal_proc - 1) / fs_proc * hop_scale;
    t_exp_raw  = (target_idx_exp_raw  - 1) / fs_raw  * hop_scale;
    t_exp_proc = (target_idx_exp_proc - 1) / fs_proc * hop_scale;

    %% --------- === Chosen channel stats === ---------
    x_cal_raw  = raw_calib(ch,:);
    x_cal_proc = proc_calib(ch,:);
    x_exp_raw  = raw_exp(ch,:);
    x_exp_proc = proc_exp(ch,:);

    S_cal_raw  = stats(x_cal_raw);
    S_cal_proc = stats(x_cal_proc);
    S_exp_raw  = stats(x_exp_raw);
    S_exp_proc = stats(x_exp_proc);

    Summary_ch.S(i)               = i;
    Summary_ch.CalRawMean(i)      = S_cal_raw.mean;
    Summary_ch.CalRawStd(i)       = S_cal_raw.std;
    Summary_ch.CalProcMean(i)     = S_cal_proc.mean;
    Summary_ch.CalProcStd(i)      = S_cal_proc.std;
    Summary_ch.CalRawClipPct(i)   = S_cal_raw.clipPct;
    Summary_ch.CalProcClipPct(i)  = S_cal_proc.clipPct;
    Summary_ch.CalRawArtPct(i)    = S_cal_raw.artPct;
    Summary_ch.CalProcArtPct(i)   = S_cal_proc.artPct;

    Summary_ch.ExpRawMean(i)      = S_exp_raw.mean;
    Summary_ch.ExpRawStd(i)       = S_exp_raw.std;
    Summary_ch.ExpProcMean(i)     = S_exp_proc.mean;
    Summary_ch.ExpProcStd(i)      = S_exp_proc.std;
    Summary_ch.ExpRawClipPct(i)   = S_exp_raw.clipPct;
    Summary_ch.ExpProcClipPct(i)  = S_exp_proc.clipPct;
    Summary_ch.ExpRawArtPct(i)    = S_exp_raw.artPct;
    Summary_ch.ExpProcArtPct(i)   = S_exp_proc.artPct;

    Summary_ch.CalFlatRaw(i)      = double(S_cal_raw.isFlat);
    Summary_ch.CalFlatProc(i)     = double(S_cal_proc.isFlat);
    Summary_ch.ExpFlatRaw(i)      = double(S_exp_raw.isFlat);
    Summary_ch.ExpFlatProc(i)     = double(S_exp_proc.isFlat);
    Summary_ch.AnyHugeRaw(i)      = double(S_cal_raw.isHuge | S_exp_raw.isHuge);
    Summary_ch.AnyHugeProc(i)     = double(S_cal_proc.isHuge | S_exp_proc.isHuge);

    %% --------- === All-channel average stats === ---------
    Ccal = size(raw_calib,1);
    Cexp = size(raw_exp,1);

    m_cal_raw  = zeros(Ccal,1);  s_cal_raw  = zeros(Ccal,1);  c_cal_raw  = zeros(Ccal,1);  a_cal_raw  = zeros(Ccal,1);  f_cal_raw  = zeros(Ccal,1); f_cal_proc  = zeros(Ccal,1); h_cal_raw  = zeros(Ccal,1); h_cal_proc  = zeros(Ccal,1);
    m_cal_proc = zeros(Ccal,1);  s_cal_proc = zeros(Ccal,1);  c_cal_proc = zeros(Ccal,1);  a_cal_proc = zeros(Ccal,1);
    m_exp_raw  = zeros(Cexp,1);  s_exp_raw  = zeros(Cexp,1);  c_exp_raw  = zeros(Cexp,1);  a_exp_raw  = zeros(Cexp,1);  f_exp_raw  = zeros(Cexp,1); f_exp_proc  = zeros(Cexp,1); h_exp_raw  = zeros(Cexp,1); h_exp_proc  = zeros(Cexp,1);
    m_exp_proc = zeros(Cexp,1);  s_exp_proc = zeros(Cexp,1);  c_exp_proc = zeros(Cexp,1);  a_exp_proc = zeros(Cexp,1);

    for c = 1:Ccal
        sc_r = stats(raw_calib(c,:));
        sc_p = stats(proc_calib(c,:));
        m_cal_raw(c)  = sc_r.mean;  s_cal_raw(c)  = sc_r.std;  c_cal_raw(c)  = sc_r.clipPct;  a_cal_raw(c) = sc_r.artPct;  f_cal_raw(c) = sc_r.isFlat; f_cal_proc(c) = sc_p.isFlat; h_cal_raw(c) = sc_r.isHuge; h_cal_proc(c) = sc_p.isHuge;
        m_cal_proc(c) = sc_p.mean;  s_cal_proc(c) = sc_p.std;  c_cal_proc(c) = sc_p.clipPct;  a_cal_proc(c) = sc_p.artPct;
    end
    for c = 1:Cexp
        se_r = stats(raw_exp(c,:));
        se_p = stats(proc_exp(c,:));
        m_exp_raw(c)  = se_r.mean;  s_exp_raw(c)  = se_r.std;  c_exp_raw(c)  = se_r.clipPct;  a_exp_raw(c) = se_r.artPct;  f_exp_raw(c) = se_r.isFlat;  f_exp_proc(c) = se_p.isFlat; h_exp_raw(c) = se_r.isHuge; h_exp_proc(c) = se_p.isHuge;
        m_exp_proc(c) = se_p.mean;  s_exp_proc(c) = se_p.std;  c_exp_proc(c) = se_p.clipPct;  a_exp_proc(c) = se_p.artPct;
    end

    Summary_all.S(i)               = i;
    Summary_all.CalRawMean(i)      = mean(m_cal_raw);
    Summary_all.CalRawStd(i)       = mean(s_cal_raw);
    Summary_all.CalProcMean(i)     = mean(m_cal_proc);
    Summary_all.CalProcStd(i)      = mean(s_cal_proc);
    Summary_all.CalRawClipPct(i)   = mean(c_cal_raw);
    Summary_all.CalProcClipPct(i)  = mean(c_cal_proc);
    Summary_all.CalRawArtPct(i)    = mean(a_cal_raw);
    Summary_all.CalProcArtPct(i)   = mean(a_cal_proc);

    Summary_all.ExpRawMean(i)      = mean(m_exp_raw);
    Summary_all.ExpRawStd(i)       = mean(s_exp_raw);
    Summary_all.ExpProcMean(i)     = mean(m_exp_proc);
    Summary_all.ExpProcStd(i)      = mean(s_exp_proc);
    Summary_all.ExpRawClipPct(i)   = mean(c_exp_raw);
    Summary_all.ExpProcClipPct(i)  = mean(c_exp_proc);
    Summary_all.ExpRawArtPct(i)    = mean(a_exp_raw);
    Summary_all.ExpProcArtPct(i)   = mean(a_exp_proc);

    Summary_all.CalFlatRaw(i)      = mean(f_cal_raw);
    Summary_all.CalFlatProc(i)     = mean(f_cal_proc);
    Summary_all.ExpFlatRaw(i)      = mean(f_exp_raw);
    Summary_all.ExpFlatProc(i)     = mean(f_exp_proc);
    Summary_all.AnyHugeRaw(i)      = mean(h_cal_raw | h_exp_raw);
    Summary_all.AnyHugeProc(i)     = mean(h_cal_proc | h_exp_proc);

    %% --------- === Plot for chosen channel === ---------
    figure(100+i); clf; set(gcf,'Color','w','Position',[80 80 1150 600]);
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

    nexttile;
    plot(t_cal_raw,  x_cal_raw(target_idx_cal_raw), 'k'); grid on;
    xlabel('Time (s)'); ylabel('\muV'); title(sprintf('S%d CALIB RAW (ch %d)', i, ch));

    nexttile;
    plot(t_cal_proc, x_cal_proc(target_idx_cal_proc), 'k'); grid on;
    xlabel('Time (s)'); ylabel('\muV'); title(sprintf('S%d CALIB PROC (ch %d)', i, ch));

    nexttile;
    plot(t_exp_raw,  x_exp_raw(target_idx_exp_raw), 'k'); grid on;
    xlabel('Time (s)'); ylabel('\muV'); title(sprintf('S%d EXP RAW (ch %d)', i, ch));

    nexttile;
    plot(t_exp_proc, x_exp_proc(target_idx_exp_proc), 'k'); grid on;
    xlabel('Time (s)'); ylabel('\muV'); title(sprintf('S%d EXP PROC (ch %d)', i, ch));
end

%% --------- Display summaries ---------
disp('=== QC Summary (all channels averaged) ===');
disp(Summary_all,2);
