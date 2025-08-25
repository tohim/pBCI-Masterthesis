import socket
import threading
import time

# Robot Connection
ROBOT_HOST = "192.168.0.61"
ROBOT_PORT = 30002

# MATLAB Connection
MATLAB_HOST = "127.0.0.1"
MATLAB_PORT = 65432

running = False         # Whether robot is actively moving
current_speed = "LOW"   # Default speed

class Scripter:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ROBOT_HOST, ROBOT_PORT))

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

    def move_task_1_a(self, speed=0.5):
        self.send_command(f"movej([1.652, -0.803, 0.924, -1.725, -1.541, 0.051], a=0.5, v={speed})\n", 6)

    def move_task_1_b(self, speed=0.4):
        self.send_command(f"movej([1.652, -0.746, 1.063, -1.920, -1.542, 0.053], a=0.5, v={speed})\n", 3)

    def move_home(self):
        self.send_command("movej([0.887, -1.469, 1.803, -1.880, -1.553, 2.331], a=0.6, v=0.9)\n", 6)

    def close(self):
        self.sock.close()

def listen_for_matlab(conn):
    global running, current_speed
    while True:
        data = conn.recv(1024).decode('utf-8').strip()
        if data == "STOP":
            print("[MATLAB] STOP received – pausing robot.")
            running = False
            conn.sendall(b"ACK: STOP")
        elif data == "CONTINUE":
            print("[MATLAB] CONTINUE received – resuming robot.")
            running = True
            conn.sendall(b"ACK: CONTINUE")
        elif data in ["LOW", "HIGH"]:
            current_speed = data
            print(f"[MATLAB] Speed profile set to: {current_speed}")
            conn.sendall(b"ACK: SPEED")
        else:
            print(f"[MATLAB] Unknown command: {data}")
            conn.sendall(b"ACK: UNKNOWN")

def main():
    global running, current_speed
    scripter = Scripter()

    # MATLAB TCP Server setup
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.bind((MATLAB_HOST, MATLAB_PORT))
        server_sock.listen(1)
        print("Waiting for MATLAB to connect…")
        conn, addr = server_sock.accept()
        print("MATLAB connected:", addr)

        # Start a thread to listen for MATLAB commands
        threading.Thread(target=listen_for_matlab, args=(conn,), daemon=True).start()

        scripter.move_home()

        # Task sequence to ensure gripper is open
        scripter.open_gripper()
        scripter.close_gripper()
        scripter.open_gripper()

        while True:
            # Determine speed based on current_speed
            if current_speed == "LOW":
                v_a, v_b = 0.4, 0.3
            else:
                v_a, v_b = 0.9, 0.8

            # Execute movement pattern
            check_pause()  # Check pause after each motion step
            scripter.move_task_1_a(speed=v_a)       # going up and back
            check_pause()
            scripter.open_gripper()
            check_pause()
            scripter.move_task_1_b(speed=v_b)       # going down
            check_pause()
            scripter.close_gripper()
            check_pause()
            scripter.move_task_1_a(speed=v_a)
            check_pause()
            scripter.move_home()
            check_pause()

def check_pause():
    global running
    while not running:
        print("Robot is paused – waiting for CONTINUE command from MATLAB…")
        time.sleep(0.1)  # Short sleep to reduce CPU usage

if __name__ == "__main__":
    main()
