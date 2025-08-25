import socket
import threading
import time

# Robot Connection
ROBOT_HOST = "192.168.0.61"
ROBOT_PORT = 30002

# MATLAB Connection
MATLAB_HOST = "127.0.0.1"
MATLAB_PORT = 65432

running = False  # Whether robot is actively moving
current_speed = "LOW"  # Default speed


class Scripter:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ROBOT_HOST, ROBOT_PORT))

    def send_command(self, cmd: str, base_delay: float = 0.0, speed: float = 0.8):
        self.sock.sendall(cmd.encode('utf-8'))

        # Add extra delay only for LOW speed
        if speed == 0.3:
            scaled_delay = base_delay + 2.5 * 0.9
        else:
            scaled_delay = base_delay

        print(f"[COMMAND] Sent: {cmd.strip()} | Delay: {scaled_delay:.2f}s")
        if scaled_delay > 0:
            time.sleep(scaled_delay)



    def open_gripper(self):
        self.send_command("set_tool_digital_out(0, False)\n", 1)
        self.send_command("set_tool_digital_out(1, True)\n", 0.5)
        self.send_command("set_tool_digital_out(1, False)\n", 0.5)

    def close_gripper(self):
        self.send_command("set_tool_digital_out(0, True)\n", 1)

    def move_old_down(self, accel, speed):
        # move to old down position
        self.send_command(
            f"movej([0.186, -0.877, 1.302, -1.981, -1.525, -1.402], a={accel}, v={speed})\n",
            base_delay=1.2, speed=speed
        )

    def move_old_up_x(self, accel, speed):
        # move to old up x position
        self.send_command(
            f"movej([0.186, -0.951, 1.179, -1.784, -1.524, -1.404], a={accel}, v={speed})\n",
            base_delay=2.2, speed=speed
        )

    def move_piece_1_down(self, accel, speed):
        # move piece 1 down
        self.send_command(
            f"movej([1.226, -1.264, 1.945, -2.273, -1.531, -0.358], a={accel}, v={speed})\n",
            base_delay=1.5, speed=speed
        )

    def move_piece_1_up_short(self, accel, speed):
        # move piece 1 up short
        self.send_command(
            f"movej([1.227, -1.432, 1.739, -1.899, -1.528, -0.362], a={accel}, v={speed})\n",
            base_delay=1.5, speed=speed
        )

    def move_piece_2_down(self, accel, speed):
        # move piece 2 down
        self.send_command(
            f"movej([0.998, -1.195, 1.842, -2.232, -1.527, -0.587], a={accel}, v={speed})\n",
            base_delay=1.5, speed=speed
        )

    def move_piece_2_up_short(self, accel, speed):
        # move piece 2 up short
        self.send_command(
            f"movej([0.998, -1.351, 1.641, -1.874, -1.523, -0.591], a={accel}, v={speed})\n",
            base_delay=1.5, speed=speed
        )

    def move_piece_3_down(self, accel, speed):
        # move piece 3 down
        self.send_command(
            f"movej([0.820, -1.079, 1.662, -2.161, -1.524, -0.766], a={accel}, v={speed})\n",
            base_delay=1.5, speed=speed
        )

    def move_piece_3_up_short(self, accel, speed):
        # move piece 3 up short
        self.send_command(
            f"movej([0.820, -1.217, 1.461, -1.822, -1.521, -0.770], a={accel}, v={speed})\n",
            base_delay=1.5, speed=speed
        )

    def move_piece_4_down(self, accel, speed):
        # move piece 4 down
        self.send_command(
            f"movej([0.686, -0.894, 1.347, -2.027, -1.523, -0.900], a={accel}, v={speed})\n",
            base_delay=1.5, speed=speed
        )

    def move_piece_4_up_short(self, accel, speed):
        # move piece 4 up short
        self.send_command(
            f"movej([0.687, -0.999, 1.136, -1.713, -1.520, -0.904], a={accel}, v={speed})\n",
            base_delay=1.5, speed=speed
        )

    def move_piece_1_old_up(self, accel, speed):
        # move piece 1 old up
        self.send_command(
            f"movej([0.186, -0.951, 1.179, -1.784, -1.524, -1.404], a={accel}, v={speed})\n",
            base_delay=4, speed=speed
        )

    def move_piece_2_old_up(self, accel, speed):
        # move piece 2 old up
        self.send_command(
            f"movej([0.186, -0.951, 1.179, -1.784, -1.524, -1.404], a={accel}, v={speed})\n",
            base_delay=3.3, speed=speed
        )

    def move_piece_3_old_up(self, accel, speed):
        # move piece 3 old up
        self.send_command(
            f"movej([0.186, -0.951, 1.179, -1.784, -1.524, -1.404], a={accel}, v={speed})\n",
            base_delay=2.6, speed=speed
        )

    def move_piece_4_old_up(self, accel, speed):
        # move piece 4 old up
        self.send_command(
            f"movej([0.186, -0.951, 1.179, -1.784, -1.524, -1.404], a={accel}, v={speed})\n",
            base_delay=2.2, speed=speed
        )

    def move_piece_1_up_from_base(self, accel, speed):
        # move piece 1 from base
        self.send_command(
            f"movej([1.227, -1.432, 1.739, -1.899, -1.528, -0.362], a={accel}, v={speed})\n",
            base_delay=4, speed=speed
        )

    def move_piece_2_up_from_base(self, accel, speed):
        # move piece 2 from base
        self.send_command(
            f"movej([0.998, -1.351, 1.641, -1.874, -1.523, -0.591], a={accel}, v={speed})\n",
            base_delay=3.3, speed=speed
        )

    def move_piece_3_up_from_base(self, accel, speed):
        # move piece 3 from base
        self.send_command(
            f"movej([0.820, -1.217, 1.461, -1.822, -1.521, -0.770], a={accel}, v={speed})\n",
            base_delay=2.6, speed=speed
        )

    def move_piece_4_up_from_base(self, accel, speed):
        # move piece 4 from base
        self.send_command(
            f"movej([0.687, -0.999, 1.136, -1.713, -1.520, -0.904], a={accel}, v={speed})\n",
            base_delay=2.2, speed=speed
        )

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

        # Task sequence to ensure gripper is open
        scripter.open_gripper()
        scripter.close_gripper()
        scripter.open_gripper()

        # Position Sequences
        piece_cycle = [
            (scripter.move_piece_1_up_from_base, scripter.move_piece_1_down, scripter.move_piece_1_up_short,
             scripter.move_piece_1_old_up),
            (scripter.move_piece_2_up_from_base, scripter.move_piece_2_down, scripter.move_piece_2_up_short,
             scripter.move_piece_2_old_up),
            (scripter.move_piece_3_up_from_base, scripter.move_piece_3_down, scripter.move_piece_3_up_short,
             scripter.move_piece_3_old_up),
            (scripter.move_piece_4_up_from_base, scripter.move_piece_4_down, scripter.move_piece_4_up_short,
             scripter.move_piece_4_old_up),
        ]

    while True:
        for from_base, down_func, up_return_func, home_func in piece_cycle:

            # Evaluate speed fresh before each command
            check_pause()
            accel, speed = (0.3, 0.3) if current_speed == "LOW" else (0.9, 0.8)
            from_base(accel, speed)

            check_pause()
            scripter.open_gripper()

            check_pause()
            accel, speed = (0.3, 0.3) if current_speed == "LOW" else (0.9, 0.8)
            down_func(accel, speed)

            check_pause()
            scripter.close_gripper()

            check_pause()
            accel, speed = (0.3, 0.3) if current_speed == "LOW" else (0.9, 0.8)
            up_return_func(accel, speed)

            check_pause()
            accel, speed = (0.3, 0.3) if current_speed == "LOW" else (0.9, 0.8)
            home_func(accel, speed)

            check_pause()
            accel, speed = (0.3, 0.3) if current_speed == "LOW" else (0.9, 0.8)
            from_base(accel, speed)

            check_pause()
            accel, speed = (0.3, 0.3) if current_speed == "LOW" else (0.9, 0.8)
            down_func(accel, speed)

            check_pause()
            scripter.open_gripper()

            check_pause()
            accel, speed = (0.3, 0.3) if current_speed == "LOW" else (0.9, 0.8)
            up_return_func(accel, speed)


def check_pause():
    global running
    while not running:
        print("Robot is paused – waiting for CONTINUE command from MATLAB…")
        time.sleep(0.1)  # Short sleep to reduce CPU usage


if __name__ == "__main__":
    main()
