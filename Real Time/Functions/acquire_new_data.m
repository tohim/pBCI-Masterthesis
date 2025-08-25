% -------------------------------------------------------------------------
% GET BUFFERED DATA FUNCTION
% -------------------------------------------------------------------------
function [eeg_segment, raw_eeg_segment] = acquire_new_data(eeg_protocol, state, tcp_server_simulink, label, simulate)

% Handle missing inputs
if nargin < 4
    label = [];
end
if nargin < 5
    simulate = [];
end

% Parameters
expected_channels = 14;    % might change to 16 because of simulink handle takes 16 channels -> also change in the simulink models!!
expected_samples = 1000;   % 4 sec @ 250 Hz
expected_size = expected_channels * expected_samples;

switch lower(state)

    case 'realtime'

        switch lower(eeg_protocol)

            case 'udp'  % -> UDP code is kinda "legacy code" - was only implemented in the very early stage
                % UDP config
                udp_port = 8844;
                timeout_sec = 2.25;  % we expect new package every 2 seconds (4sec epochs 50% overlap) -> slight overhead for timeout

                % Handle Retry and Reconnect
                max_retries = 10;
                retries = 0;

                % Persistent Variable to avoid Reloading udp_rx every time we call
                % the acquire_new_data function
                persistent udp_rx   % udp_rx = UDP object & rx = receive -> "UDP receiver object"

                if isempty(udp_rx)                                                          % Waiting until new data arrives
                    fprintf('[UDP] Initializing EEG listener on port %d...\n', udp_port);
                    udp_rx = udpport("datagram", "IPV4", "LocalPort", udp_port);            % Initialize UDP port object "udp_rx"
                end

                while true
                    try
                        % Clear older packets (latest only mode) -> Read single complete (,1,) incoming UDP datagram
                        while udp_rx.NumDatagramsAvailable > 1                              % Only take the most current incoming data
                            read(udp_rx, 1, "double");
                        end

                        % Try to read 1 frame (reading the udpport = data coming from simulink)
                        % reads udp packet into a vector of doubles (raw_data)
                        % Matlab waits for 1 sec for a packet before throwing error
                        raw_data = read(udp_rx, 1, "double", Timeout=timeout_sec);

                        if numel(raw_data) == expected_size
                            eeg_segment = reshape(raw_data, expected_channels, expected_samples);
                            fprintf('[UDP] Received EEG | Size: %s | Timestamp: %.3f s\n', mat2str(size(eeg_segment)), toc);

                            % Downsample incoming eeg to 128 Hz
                            fs_old = 250;
                            fs_new = 128;

                            % Downsamples using resample (applies anti-aliasing FIR filter)
                            eeg_segment = resample(eeg_segment', fs_new, fs_old)';

                            break;  % = got a good packet, exit loop
                        else
                            warning('[UDP] Bad packet (size: %d), waiting for valid EEG...', numel(raw_data));
                        end

                        % If Bad packet, wait for new good EEG data
                    catch
                        retries = retries + 1;
                        fprintf('[UDP] Waiting for EEG data... (%d)\n', retries);

                        if retries >= max_retries
                            warning('[UDP] No data after %d attempts. Reinitializing UDP connection...', max_retries);
                            try
                                clear udp_rx  % Remove stale connection
                            catch
                                % catch, in case udp_rx is already invalid, to not
                                % crash code
                            end
                            udp_rx = udpport("datagram", "IPV4", "LocalPort", udp_port);  % Recreate Connection
                            retries = 0;
                            pause(1);  % Small delay before retrying
                        else
                            pause(0.2);
                        end
                    end
                end

            case 'tcp'

                % TCP logic is basically mirroring the UDP logic + some
                % additional checks and flushing

                % TCP config
                timeout_sec = 2.25;  % reasonable timeout (like UDP)
                max_retries = 15;
                retries = 0;

                while true
                    try
                        % Check if data available
                        % before check if too many available (if there was some code execution/ system): -> Flush TCP buffer
                        while tcp_server_simulink.NumBytesAvailable > expected_size * 8   % 8 bytes per double
                            bytes_to_flush = tcp_server_simulink.NumBytesAvailable - expected_size * 8;
                            if bytes_to_flush > 0
                                read(tcp_server_simulink, bytes_to_flush, "uint8");
                                fprintf('[TCP] Flushed %d bytes of stale data.\n', bytes_to_flush);
                            end
                        end

                        % Check if enough data is available for 1 complete epoch
                        if tcp_server_simulink.NumBytesAvailable >= expected_size * 8  % 8 bytes per double

                            % Initialize read buffer
                            raw_bytes = zeros(expected_size*8, 1, "uint8");
                            bytes_read = 0;

                            % Step 1: Read in loop until expected_size * 8 bytes collected
                            while bytes_read < expected_size * 8
                                n_available = tcp_server_simulink.NumBytesAvailable;
                                n_needed = (expected_size * 8) - bytes_read;
                                n_read = min(n_available, n_needed);

                                if n_read > 0
                                    % Read as bytes, typecast to doubles
                                    % rn using little-endian - if weird numbers: use "swapbytes" on "partial" vector or "big endian" in simulink tcp block
                                    partial = read(tcp_server_simulink, n_read, "uint8");
                                    raw_bytes(bytes_read + 1 : bytes_read + n_read) = partial;
                                    bytes_read = bytes_read + n_read;
                                else
                                    pause(0.05);  % Small wait for more data
                                end
                            end

                            % Step 2: Convert entire byte buffer to double
                            raw_data = typecast(raw_bytes, "double");

                            % Step 3: Validate and reshape
                            if numel(raw_data) == expected_size
                                eeg_segment = reshape(raw_data, expected_channels, expected_samples);
                                % fprintf('[TCP] Received EEG | Size: %s | Timestamp: %.3f s\n', mat2str(size(eeg_segment)), toc);
                                
                                % Save the truly raw data before resampling is applied
                                raw_eeg_segment = eeg_segment;

                                % Downsample incoming eeg to 128 Hz
                                fs_old = 250;
                                fs_new = 128;

                                % Downsamples using resample (applies anti-aliasing FIR filter - with high pass <.5 - zero-centers signal already mostly)
                                eeg_segment = resample(eeg_segment', fs_new, fs_old)';

                                break;  % Valid EEG segment ready, exit loop
                            else
                                warning('[TCP] Incomplete packet (size: %d), waiting for valid EEG...', numel(raw_data));
                                warning('[TCP] Unexpected EEG size. Got %d values, expected %d. Retrying...', numel(raw_data), expected_size);
                            end

                        else
                            % Not enough data yet, wait a bit
                            retries = retries + 1;
                            %fprintf('[TCP] Waiting for EEG data... (%d)\n', retries);

                            if retries >= max_retries
                                warning('[TCP] No data after %d attempts. Reinitializing TCP connection...', max_retries);
                                try
                                    clear tcp_server_simulink
                                catch
                                    % Already invalid
                                end
                                % Reconnect to tcp server
                                tcp_server_simulink = tcpserver(tcp_port_simulink, ...
                                    "ConnectionChangedFcn", @(src, evt) fprintf('[TCP] Connection: %s\n', evt.Connected));
                                retries = 0;
                                pause(1);  % Wait before retry
                            else
                                pause(0.2);
                            end
                        end

                    catch ME
                        warning('[TCP] Error while reading data');
                        retries = retries + 1;
                        pause(0.2);

                        if retries >= max_retries
                            warning('[TCP] Reinitializing TCP server...');
                            try
                                clear tcp_server_simulink
                            catch
                            end
                            tcp_server_simulink = tcpserver(50001, ...
                                "ConnectionChangedFcn", @(src, evt) fprintf('[TCP] Connection: %s\n', evt.Connected));
                            retries = 0;
                            pause(1);
                        end
                    end
                end

            case 'lsl'

                % LSL already gives double precision data
                % Unbuffering is done via inlet.pull_chunk() -> gives raw
                % channels x samples
                % Circular buffer keeps a rooling history for overlapping
                % 4sec windwos - 2sec new chunks (50% overlap) fill fresh
                % data into buffer
                % Buffer & Reshape is already handled by direct
                % channel-wise buffering and final eeg_segment extraction

                % LSL config
                expected_channels = 14; 
                fs = 128;               
                epoch_sec = 4;
                epoch_length = epoch_sec * fs; 
                expected_size = expected_channels * epoch_length;

                % LSL buffering parameters
                % Using a circular buffer for rolling data (overlapping 4-sec epochs)
                persistent inlet buffer buffer_idx

                % Initialize LSL stream & buffer
                if isempty(inlet)
                    fprintf('[LSL] Resolving EEG stream...\n');
                    lib = lsl_loadlib();
                    info = lsl_resolve_byprop(lib,'type','EEG');  % Discover EEG streams
                    inlet = lsl_inlet(info{1});
                    fprintf('[LSL] EEG stream found. Ready.\n');

                    buffer = zeros(expected_channels, expected_size * 2);  % Buffer for up to 8 sec (to ensure overlap handling)
                    buffer_idx = 1;
                end

                % Collect enough new data to fill next 2-sec segment (50% overlap)
                overlap_sec = 2;
                overlap_samples = overlap_sec * fs; % samples needed for overlap in buffer to get new 4sec epoch

                % Wait for at least 2 sec of new data
                while true
                    [chunk, timestamps] = inlet.pull_chunk();
                    if isempty(chunk)
                        pause(0.05);
                        continue;
                    end

                    % Chunk: (channels x samples)
                    if size(chunk, 1) ~= expected_channels
                        chunk = chunk';  % Transpose if needed
                    end

                    % Store into buffer (circular buffer logic)
                    samples_in_chunk = size(chunk, 2);
                    idx_range = buffer_idx : buffer_idx + samples_in_chunk - 1;
                    idx_range = mod(idx_range - 1, size(buffer, 2)) + 1;  % wrap around
                    buffer(:, idx_range) = chunk;
                    buffer_idx = mod(buffer_idx - 1 + samples_in_chunk, size(buffer, 2)) + 1;

                    if samples_in_chunk >= overlap_samples
                        break;  % enough new data for next overlap
                    end
                end

                % Extract final 4-sec window (most recent 512 samples)
                idx_range_final = buffer_idx - expected_size : buffer_idx - 1;
                idx_range_final = mod(idx_range_final - 1, size(buffer, 2)) + 1;
                eeg_segment = buffer(:, idx_range_final);

                % Validate size
                if size(eeg_segment, 2) ~= expected_samples
                    warning('[LSL] Epoch size mismatch: %s', mat2str(size(eeg_segment)));
                end

                fprintf('[LSL] Received EEG | Size: %s | Timestamp: %.3f s\n', mat2str(size(eeg_segment)), toc);
        end


    case 'testing'

        % Persistent Variables to avoid Reloading every time we call
        % the acquire_new_data function - important to keep the idx_low and
        % idx_high counts when simulating new incoming eeg segments
        % same for the raw STEW eeg data files
        persistent test_data_calib test_data_rt test_labels_calib test_labels_rt test_data_low_calib ...
            test_data_low_rt test_data_high_calib test_data_high_rt idx_low_calib idx_low_rt idx_high_calib idx_high_rt

        switch lower(simulate)

            case 'calib'

                if isempty(test_data_calib)
                    fprintf('[TEST] Loading testing EEG data from STEW...\n');
                    base = 'E:\SchuleJobAusbildung\HTW\MasterThesis\Code\TrainingDatasets\Workload\STEW Dataset';
                    files = {
                        %                         fullfile(base, 'sub02_hi.txt'), ...
                        %                         fullfile(base, 'sub02_lo.txt'), ...
                        %                         fullfile(base, 'sub05_hi.txt'), ...
                        %                         fullfile(base, 'sub05_lo.txt'),...
                        %                         fullfile(base, 'sub09_hi.txt'), ...
                        %                         fullfile(base, 'sub09_lo.txt')
                        fullfile(base, 'sub04_hi.txt'), ...
                        fullfile(base, 'sub04_lo.txt'), ...
                        fullfile(base, 'sub09_hi.txt'), ...
                        fullfile(base, 'sub09_lo.txt'),...
                        fullfile(base, 'sub24_hi.txt'), ...
                        fullfile(base, 'sub24_lo.txt')
                        };

                    [test_data_calib, test_labels_calib] = load_testing_eeg_data(files, 128, 4, 0.5);

                    % Split by label
                    test_data_low_calib = test_data_calib(:,:,test_labels_calib == 0);
                    test_data_high_calib = test_data_calib(:,:,test_labels_calib == 1);
                    idx_low_calib = 1;
                    idx_high_calib = 1;
                end

                if label == 0  % LOW
                    if idx_low_calib <= size(test_data_low_calib,3)
                        eeg_segment = test_data_low_calib(:,:,idx_low_calib);
                        idx_low_calib = idx_low_calib + 1;
                    else
                        warning('[TEST] No more LOW segments!');
                        eeg_segment = zeros(14, 512);
                    end
                else  % HIGH
                    if idx_high_calib <= size(test_data_high_calib,3)
                        eeg_segment = test_data_high_calib(:,:,idx_high_calib);
                        idx_high_calib = idx_high_calib + 1;
                    else
                        warning('[TEST] No more HIGH segments!');
                        eeg_segment = zeros(14, 512);
                    end
                end

            case 'rt'

                if isempty(test_data_rt)
                    fprintf('[TEST] Loading testing EEG data from STEW...\n');
                    base = 'E:\SchuleJobAusbildung\HTW\MasterThesis\Code\TrainingDatasets\Workload\STEW Dataset';
                    files = {
                        %                         fullfile(base, 'sub01_hi.txt'), ...
                        %                         fullfile(base, 'sub01_lo.txt'), ...
                        %                         fullfile(base, 'sub22_hi.txt'), ...
                        %                         fullfile(base, 'sub22_lo.txt'),...
                        %                         fullfile(base, 'sub47_hi.txt'), ...
                        %                         fullfile(base, 'sub47_lo.txt')
                        fullfile(base, 'sub10_hi.txt'), ...
                        fullfile(base, 'sub10_lo.txt'), ...
                        fullfile(base, 'sub34_hi.txt'), ...
                        fullfile(base, 'sub34_lo.txt'),...
                        fullfile(base, 'sub45_hi.txt'), ...
                        fullfile(base, 'sub45_lo.txt')
                        };

                    [test_data_rt, test_labels_rt] = load_testing_eeg_data(files, 128, 4, 0.5);

                    % Split by label
                    test_data_low_rt = test_data_rt(:,:,test_labels_rt == 0);
                    test_data_high_rt = test_data_rt(:,:,test_labels_rt == 1);
                    idx_low_rt = 1;
                    idx_high_rt = 1;
                end

                if label == 0  % LOW
                    if idx_low_rt <= size(test_data_low_rt,3)
                        eeg_segment = test_data_low_rt(:,:,idx_low_rt);
                        idx_low_rt = idx_low_rt + 1;
                    else
                        warning('[TEST] No more LOW segments!');
                        eeg_segment = zeros(14, 512);
                    end
                else  % HIGH
                    if idx_high_rt <= size(test_data_high_rt,3)
                        eeg_segment = test_data_high_rt(:,:,idx_high_rt);
                        idx_high_rt = idx_high_rt + 1;
                    else
                        warning('[TEST] No more HIGH segments!');
                        eeg_segment = zeros(14, 512);
                    end
                end
        end
end
