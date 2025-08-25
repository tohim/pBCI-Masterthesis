function wait_for_enter()
    disp('Press ENTER to continue...');
    while true
        pause(0.1);
        ch = get(gcf, 'CurrentCharacter');
        if strcmp(ch, char(13))  % ASCII 13 = Enter
            set(gcf, 'CurrentCharacter', char(0));
            break;
        end
    end
end

