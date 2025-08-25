import socket
import time
import winsound
import msvcrt

HOST = "192.168.29.61"
PORT = 30002


# User configuration
task_a_repetitions = 2   # run Task 1 this many times
task_b_repetitions = 2   # run Task 2 this many times
task_c_repetitions = 2   # run Task 3 this many times
task_d_repetitions = 2   # run Task 4 this many times
task_e_repetitions = 2   # run Task 5 this many times
task_f_repetitions = 2   # run Task 6 this many times

def wait_for_enter(prompt: str):
    # prompt the user

    print(prompt, end='', flush=True)
    # busy–wait until they press Enter
    while True:
        if msvcrt.kbhit() and msvcrt.getwch() == '\r':
            break
    # stop the looping sound immediately
    winsound.PlaySound(None, winsound.SND_PURGE)
    print()  # move to the next line

class Scripter:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((HOST, PORT))

    def send_command(self, cmd: str, delay: float = 0.0):
        self.sock.sendall(cmd.encode('utf-8'))
        if delay:
            time.sleep(delay)

    def open_gripper(self):
        self.send_command("set_tool_digital_out(0, False)\n", 1)
        self.send_command("set_tool_digital_out(1, True)\n", 0.5)
        self.send_command("set_tool_digital_out(1, False)\n", 0.5)


    def close_gripper(self):
        self.send_command("set_tool_digital_out(0, True)\n", 1)

    # task 1 waypoints
    def move_task_1_a(self):
        self.send_command("movej([1.652, -0.803, 0.924, -1.725, -1.541, 0.051], a=0.5, v=0.5)\n",6)  # up 1

    def move_task_1_b(self):
        self.send_command("movej([1.652, -0.746, 1.063, -1.920, -1.542, 0.053], a=0.5, v=0.4)\n",3)  # down 1

    # task 2 waypoints
    def move_task_2_a(self):
        self.send_command("movej([1.669, -1.136, 1.465, -1.934, -1.542, 0.071], a=0.5, v=0.5)\n",6)  # up 2

    def move_task_2_b(self):
        self.send_command("movej([1.664, -1.053, 1.617, -2.168, -1.544, 0.068], a=0.5, v=0.4)\n",3)  # down 2

    # task 3 waypoints
    def move_task_3_a(self):
        self.send_command("movej([1.689, -1.443, 1.871, -2.033, -1.543, 0.092], a=0.5, v=0.5)\n",6)  # up 3

    def move_task_3_b(self):
        self.send_command("movej([1.688, -1.311, 2.002, -2.296, -1.546, 0.094], a=0.5, v=0.4)\n",3)  # down 3

    # task 4 waypoints
    def move_task_4_a(self):
        self.send_command("movej([1.453, -0.886, 1.032, -1.740, -1.529, -0.099], a=0.5, v=0.5)\n",6)  # up 4

    def move_task_4_b(self):
        self.send_command("movej([1.456, -0.812, 1.187, -1.968, -1.530, -0.093], a=0.4, v=0.4)\n",3)  # down 4

    # task 5 waypoints
    def move_task_5_a(self):
        self.send_command("movej([1.431, -1.197, 1.525, -1.926, -1.535, -0.168], a=0.5, v=0.5)\n",5)  # up 5

    def move_task_5_b(self):
        self.send_command("movej([1.430, -1.086, 1.663, -2.176, -1.537, -0.165], a=0.5, v=0.4)\n",3)  # down 5

    # task 6 waypoints
    def move_task_6_a(self):
        self.send_command("movej([1.394, -1.485, 1.911, -2.025, -1.534, -0.204], a=0.5, v=0.5)\n",5)  # up 6

    def move_task_6_b(self):
        self.send_command("movej([1.393, -1.344, 2.050, -2.305, -1.537, -0.201], a=0.5, v=0.4)\n",3)  # down 6


    # Home positions
    def move_home(self):
        self.send_command("movej([0.887, -1.469, 1.803, -1.880, -1.553, 2.331], a=0.6, v=0.9)\n", 6)
    def move_home_after_task(self):
        self.send_command("movej([0.887, -1.469, 1.803, -1.880, -1.553, 2.331], a=0.6, v=0.9)\n", 5)

    def close(self):
        self.sock.close()


def task1(s: Scripter):
    s.open_gripper()
    s.move_task_1_a()
    wait_for_enter("Task 1 on hold; press Enter to continue…")
    s.move_task_1_b()
    s.close_gripper()
    s.move_task_1_a()
    s.move_home_after_task()
    s.move_task_1_a()
    s.move_task_1_b()
    s.open_gripper()
    s.move_task_1_a()

def task2(s: Scripter):
    s.open_gripper()
    s.move_task_2_a()
    wait_for_enter("Task 2 on hold; press Enter to continue…")
    s.move_task_2_b()
    s.close_gripper()
    s.move_task_2_a()
    s.move_home_after_task()
    s.move_task_2_a()
    s.move_task_2_b()
    s.open_gripper()
    s.move_task_2_a()

def task3(s: Scripter):
    s.open_gripper()
    s.move_task_3_a()
    wait_for_enter("Task 3 on hold; press Enter to continue…")
    s.move_task_3_b()
    s.close_gripper()
    s.move_task_3_a()
    s.move_home_after_task()
    s.move_task_3_a()
    s.move_task_3_b()
    s.open_gripper()
    s.move_task_3_a()

def task4(s: Scripter):
    s.open_gripper()
    s.move_task_4_a()
    wait_for_enter("Task 4 on hold; press Enter to continue…")
    s.move_task_4_b()
    s.close_gripper()
    s.move_task_4_a()
    s.move_home_after_task()
    s.move_task_4_a()
    s.move_task_4_b()
    s.open_gripper()
    s.move_task_4_a()

def task5(s: Scripter):
    s.open_gripper()
    s.move_task_5_a()
    wait_for_enter("Task 5 on hold; press Enter to continue…")
    s.move_task_5_b()
    s.close_gripper()
    s.move_task_5_a()
    s.move_home_after_task()
    s.move_task_5_a()
    s.move_task_5_b()
    s.open_gripper()
    s.move_task_5_a()

def task6(s: Scripter):
    s.open_gripper()
    s.move_task_6_a()
    wait_for_enter("Task 6 on hold; press Enter to continue…")
    s.move_task_6_b()
    s.close_gripper()
    s.move_task_6_a()
    s.move_home_after_task()
    s.move_task_6_a()
    s.move_task_6_b()
    s.open_gripper()
    s.move_task_6_a()


def main():
    s = Scripter()
    s.open_gripper()
    s.close_gripper()
    s.open_gripper()
    s.move_home()

    # run each task the configured number of times
    for i in range(task_a_repetitions):
        print(f"Running Task 1 iteration {i+1}/{task_a_repetitions}")
        task1(s)

    for i in range(task_b_repetitions):
        print(f"Running Task 2 iteration {i+1}/{task_b_repetitions}")
        task2(s)

    for i in range(task_c_repetitions):
        print(f"Running Task 3 iteration {i+1}/{task_c_repetitions}")
        task3(s)

    for i in range(task_d_repetitions):
        print(f"Running Task 4 iteration {i+1}/{task_d_repetitions}")
        task4(s)

    for i in range(task_e_repetitions):
        print(f"Running Task 5 iteration {i+1}/{task_e_repetitions}")
        task5(s)

    for i in range(task_f_repetitions):
        print(f"Running Task 6 iteration {i+1}/{task_f_repetitions}")
        task6(s)

    print("All tasks complete. Returning home.")
    s.move_home()
    s.close()


if __name__ == "__main__":
    main()