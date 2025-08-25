import socket
from time import sleep
import csv
import os

# Robot Connection
HOST = "192.168.29.61"
PORT = 30002

# Matlab Connection
MATLAB_HOST = "127.0.0.1"
MATLAB_PORT = 65432

class Scripter:

    def __init__(self):
        self.script_commands = []
        self.last_movement = None
        self.movements = []
        self.delays = []

    def open_gripper(self):
        # Set digital output 0 to False
        self.send_command("set_tool_digital_out(0, False)\n", 0.5)
    
        # Set digital output 1 to True
        self.send_command("set_tool_digital_out(1, True)\n", 0.5)
    
        # Set digital output 1 to False
        self.send_command("set_tool_digital_out(1, False)\n", 0.5)

    def close_gripper(self):
        self.send_command("set_tool_digital_out(0, True)\n", 1) # close 0 true

    def move_check_up(self):
        self.send_command("movej([0.873, -1.134, 1.414, -1.868, -1.518, -0.716], a=0.5, v=0.5)\n", 3.5) # Up

    def move_check_down(self):
        self.send_command("movej([0.873, -1.061, 1.532, -2.059, -1.519, -0.714], a=0.5, v=0.5)\n", 2.5) # Down

    def move_piece_1_up(self): # Up 1
        self.send_command("movej([1.652, -0.803, 0.924, -1.725, -1.541, 0.051], a=0.5, v=0.5)\n", 6) # Up 1

    def move_piece_1_down(self): # Down 1
        self.send_command("movej([1.652, -0.743, 1.066, -1.927, -1.542, 0.053], a=0.5, v=0.4)\n", 3) # Down 1

    def move_piece_2_up(self): # Up 2
        self.send_command("movej([1.669, -1.136, 1.465, -1.934, -1.542, 0.071], a=0.5, v=0.5)\n", 6) # Up 2

    def move_piece_2_down(self): # Down 2
        self.send_command("movej([1.664, -1.053, 1.617, -2.168, -1.544, 0.068], a=0.5, v=0.4)\n", 3) # Down 2

    def move_piece_3_up(self): # Up 3
        self.send_command("movej([1.689, -1.443, 1.871, -2.033, -1.543, 0.092], a=0.5, v=0.5)\n", 6) # Up 3

    def move_piece_3_down(self): # Down 3
        self.send_command("movej([1.688, -1.311, 2.002, -2.296, -1.546, 0.094], a=0.5, v=0.4)\n", 3) # Down 3

    def move_piece_4_up(self): # Up 4
        self.send_command("movej([1.453, -0.886, 1.032, -1.740, -1.529, -0.099], a=0.5, v=0.5)\n", 6) # Up 4

    def move_piece_4_down(self): # Down 4
        self.send_command("movej([1.456, -0.812, 1.187, -1.968, -1.530, -0.093], a=0.4, v=0.4)\n", 3) # Down 4

    def move_piece_5_up(self): # Up 5
        self.send_command("movej([1.431, -1.197, 1.525, -1.926, -1.535, -0.168], a=0.5, v=0.5)\n", 5) # Up 5

    def move_piece_5_down(self): # Down 5
        self.send_command("movej([1.430, -1.086, 1.663, -2.176, -1.537, -0.165], a=0.5, v=0.4)\n", 3) # Down 5

    def move_piece_6_up(self): # Up 6
        self.send_command("movej([1.377, -1.517, 2.072, -2.154, -1.535, -0.201], a=0.5, v=0.5)\n", 5) # Up 6

    def move_piece_6_down(self): # Down 6
        self.send_command("movej([1.376, -1.421, 2.148, -2.326, -1.537, -0.200], a=0.5, v=0.4)\n", 3) # Down 6

    def send_command(self, command, delay):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((HOST, PORT))
            sock.send(command.encode('utf-8'))
        sleep(delay)  # Allow time for the robot to execute the command

    def save_to_csv(self, filename="delays.csv"):
        file_exists = os.path.isfile(filename)
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Run', 'Time'])
            run_number = len([row for row in csv.reader(open(filename)) if row and 'Run' in row[0]]) + 1
            writer.writerow([f'Run {run_number}'])
            for i, delay in enumerate(self.delays):
                writer.writerow([f'Time {i+1}', delay])

def run():
    for run_count in range(3): # Each run last for 4 min
        print(f"Sequence {run_count + 1}")
        scripter = Scripter()
        
        for i in range(1):
            # Task sequence
            scripter.open_gripper() 
            scripter.close_gripper() 
            scripter.open_gripper() 

            # Task 1

            scripter.move_piece_2_up() 
            scripter.open_gripper()
            scripter.move_piece_2_down() 
            scripter.close_gripper() 
            scripter.move_piece_2_up() 
            scripter.move_check_up() 
            scripter.move_check_down() 
            scripter.move_check_up() 

            scripter.move_piece_2_up() 
            scripter.move_piece_2_down() 
            scripter.open_gripper() 
            scripter.move_piece_2_up()

            # Task 2  

            scripter.move_piece_1_up() 
            scripter.open_gripper()
            scripter.move_piece_1_down() 
            scripter.close_gripper() 
            scripter.move_piece_1_up() 
            scripter.move_check_up() 
            scripter.move_check_down() 
            scripter.move_check_up() 

            scripter.move_piece_1_up() 
            scripter.move_piece_1_down() 
            scripter.open_gripper() 
            scripter.move_piece_1_up()

            # Task 3

            scripter.move_piece_3_up() 
            scripter.open_gripper()
            scripter.move_piece_3_down() 
            scripter.close_gripper() 
            scripter.move_piece_3_up() 
            scripter.move_check_up() 
            scripter.move_check_down() 
            scripter.move_check_up() 

            scripter.move_piece_3_up() 
            scripter.move_piece_3_down() 
            scripter.open_gripper() 
            scripter.move_piece_3_up() 

            # Task 4

            scripter.move_piece_4_up() 
            scripter.open_gripper()
            scripter.move_piece_4_down() 
            scripter.close_gripper() 
            scripter.move_piece_4_up() 
            scripter.move_check_up() 
            scripter.move_check_down() 
            scripter.move_check_up() 

            scripter.move_piece_4_up() 
            scripter.move_piece_4_down() 
            scripter.open_gripper() 
            scripter.move_piece_4_up() 

            # Task 5

            scripter.move_piece_5_up() 
            scripter.open_gripper()
            scripter.move_piece_5_down() 
            scripter.close_gripper() 
            scripter.move_piece_5_up() 
            scripter.move_check_up() 
            scripter.move_check_down() 
            scripter.move_check_up() 

            scripter.move_piece_5_up() 
            scripter.move_piece_5_down() 
            scripter.open_gripper() 
            scripter.move_piece_5_up() 

            # Task 6

            scripter.move_piece_6_up() 
            scripter.open_gripper()
            scripter.move_piece_6_down() 
            scripter.close_gripper() 
            scripter.move_piece_6_up() 
            scripter.move_check_up() 
            scripter.move_check_down() 
            scripter.move_check_up() 

            scripter.move_piece_6_up() 
            scripter.move_piece_6_down() 
            scripter.open_gripper() 
            scripter.move_piece_6_up()

            scripter.move_check_up() 
            scripter.move_check_down() 
            scripter.move_check_up() 

    # scripter.save_to_csv()

if __name__ == "__main__":
    run()
