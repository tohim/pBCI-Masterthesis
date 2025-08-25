function send_robot_command(robot_protocol, robot_handle, command)

% Sends a UDP command to the robot arm.
fprintf('Sending Command to Robot: %s\n', command);

if strcmpi(robot_protocol, 'udp')
    writeline(robot_handle, command);

elseif strcmpi(robot_protocol, 'tcp')
    write(robot_handle, uint8([command, newline]));  % TCP often needs newline
    
end

end

