import socket
import time

HOST = "192.168.0.61"
PORT = 30002

class Scripter:
    def __init__(self):
        # connect to robot controller
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((HOST, PORT))

    def send_command(self, cmd: str, delay: float = 0.0):
        # send a URScript command and optionally wait
        self.sock.sendall(cmd.encode('utf-8'))
        if delay:
            time.sleep(delay)

    def open_gripper(self):
        # set digital output 0 to False
        self.send_command("set_tool_digital_out(0, False)\n", 0.4)
        # set digital output 1 to True
        self.send_command("set_tool_digital_out(1, True)\n", 0.5)
        # set digital output 1 to False
        self.send_command("set_tool_digital_out(1, False)\n", 0.4)

    def close_gripper(self):
        # set digital output 0 to True
        self.send_command("set_tool_digital_out(0, True)\n", 0.4)

    def move_old_down(self):
        # move to old down position
        self.send_command(
            "movej([0.186, -0.877, 1.302, -1.981, -1.525, -1.402], a=0.8, v=0.8)\n",
            1.5
        )

    def move_old_up_x(self):
        # move to old up x position
        self.send_command(
            "movej([0.186, -0.951, 1.179, -1.784, -1.524, -1.404], a=0.8, v=0.8)\n",
            3.5
        )

    def move_piece_1_old_up(self):
        # move piece 1 old up
        self.send_command(
            "movej([0.186, -0.951, 1.179, -1.784, -1.524, -1.404], a=0.8, v=0.8)\n",
            4
        )

    def move_piece_2_old_up(self):
        # move piece 2 old up
        self.send_command(
            "movej([0.186, -0.951, 1.179, -1.784, -1.524, -1.404], a=0.8, v=0.8)\n",
            3.3
        )

    def move_piece_3_old_up(self):
        # move piece 3 old up
        self.send_command(
            "movej([0.186, -0.951, 1.179, -1.784, -1.524, -1.404], a=0.8, v=0.8)\n",
            2.6
        )

    def move_piece_4_old_up(self):
        # move piece 4 old up
        self.send_command(
            "movej([0.186, -0.951, 1.179, -1.784, -1.524, -1.404], a=0.8, v=0.8)\n",
            2.2
        )

    def move_middle_point(self):
        # move to middle point
        self.send_command(
            "movej([0.391, -1.831, 2.106, -1.838, -1.521, -1.200], a=0.8, v=0.8)\n",
            3.5
        )

    def move_piece_1_down(self):
        # move piece 1 down
        self.send_command(
            "movej([1.226, -1.264, 1.945, -2.273, -1.531, -0.358], a=0.8, v=0.8)\n",
            1.5
        )

    def move_piece_1_up_test(self):
        # move piece 1 up test
        self.send_command(
            "movej([1.227, -1.432, 1.739, -1.899, -1.528, -0.362], a=0.8, v=0.8)\n",
            2.6
        )

    def move_piece_1_up_short(self):
        # move piece 1 up short
        self.send_command(
            "movej([1.227, -1.432, 1.739, -1.899, -1.528, -0.362], a=0.8, v=0.8)\n",
            1.5
        )

    def move_piece_2_down(self):
        # move piece 2 down
        self.send_command(
            "movej([0.998, -1.195, 1.842, -2.232, -1.527, -0.587], a=0.8, v=0.8)\n",
            1.6
        )

    def move_piece_2_up_test(self):
        # move piece 2 up test
        self.send_command(
            "movej([0.998, -1.351, 1.641, -1.874, -1.523, -0.591], a=0.8, v=0.8)\n",
            3.4
        )

    def move_piece_2_up_short(self):
        # move piece 2 up short
        self.send_command(
            "movej([0.998, -1.351, 1.641, -1.874, -1.523, -0.591], a=0.8, v=0.8)\n",
            1.5
        )

    def move_piece_3_down(self):
        # move piece 3 down
        self.send_command(
            "movej([0.820, -1.079, 1.662, -2.161, -1.524, -0.766], a=0.8, v=0.8)\n",
            1.5
        )

    def move_piece_3_up_test(self):
        # move piece 3 up test
        self.send_command(
            "movej([0.820, -1.217, 1.461, -1.822, -1.521, -0.770], a=0.8, v=0.8)\n",
            4.7
        )

    def move_piece_3_up_short(self):
        # move piece 3 up short
        self.send_command(
            "movej([0.820, -1.217, 1.461, -1.822, -1.521, -0.770], a=0.8, v=0.8)\n",
            1.5
        )

    def move_piece_4_down(self):
        # move piece 4 down
        self.send_command(
            "movej([0.686, -0.894, 1.347, -2.027, -1.523, -0.900], a=0.8, v=0.8)\n",
            1.5
        )

    def move_piece_4_up_test(self):
        # move piece 4 up test
        self.send_command(
            "movej([0.687, -0.999, 1.136, -1.713, -1.520, -0.904], a=0.8, v=0.8)\n",
            5.3
        )

    def move_piece_4_up_short(self):
        # move piece 4 up short
        self.send_command(
            "movej([0.687, -0.999, 1.136, -1.713, -1.520, -0.904], a=0.8, v=0.8)\n",
            1.5
        )

    def close(self):
        # clean up socket
        self.sock.close()

def main():
    s = Scripter()
    #s.move_old_up_x()
    s.move_middle_point()

    #s.open_gripper()
    #s.close_gripper()

    ################ point 1 ################
    #s.move_piece_4_up_test()
    #s.move_piece_4_down()
    #s.open_gripper()
    #s.close_gripper()
    #s.move_piece_4_up_short()
    #s.move_piece_4_old_up()
    #s.move_old_down()
    #s.open_gripper()
    #s.move_old_up_x()

    s.close()

if __name__ == "__main__":
    main()
